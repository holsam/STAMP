'''
STAMP: consensus picking logic
'''

# Import external dependencies
import json, numpy as np
from dataclasses import dataclass
from pathlib import Path
from skimage import measure

# Import internal STAMP objects
from stamp.backends.base import ToolCommand
from stamp.picking.consensus import build_particle_set, reconcile_picks_for_tomograms
from stamp.picking.geometry import extract_surface
from stamp.picking.importers import apply_prior, infer_orientations, map_tomogram_ids, read_picks
from stamp.picking.vesicles import load_vesicle_labels, summarise_vesicles, vesicle_surface_area_angstrom2
from stamp.run.state import stage_dir
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.particles import OrientationPrior
from stamp.schemas.picks import RawPick
from stamp.utils.checkpoint import file_signature, signatures, sync_checkpoint
from stamp.utils.errors import StampPipelineError, StampValidationError
from stamp.utils.io import archive_and_remove_directory, cache_tomogram_picks, glob_mrc, load_cached_picks, load_tomogram_manifests, read_mrc, write_sidecar
from stamp.utils.log import log
from stamp.utils.parallel import run_parallel
from stamp.utils.plotting.picks import plot_positions
from stamp.utils.reporting import report_beam_angle_distribution

# _import_picker: read one picks file, infer orientations per tomogram, cache under cache_dir and return the cached picks
def _import_picker(
    path: Path,
    name: str,
    manifests: list[TomogramManifest],
    vesicle_labels_by_tomogram: dict[str, Path],
    max_surface_distance_angstrom: float,
    cache_dir: Path,
    prior: OrientationPrior = OrientationPrior.NORMAL,
) -> list[RawPick]:
    picks_by_tomogram: dict[str, list[RawPick]] = {}
    for pick in map_tomogram_ids(apply_prior(read_picks(path, source_picker=name), prior), manifests):
        picks_by_tomogram.setdefault(pick.tomogram_id, []).append(pick)
    manifest_by_id = {manifest.tomogram_id: manifest for manifest in manifests}

    # a tomogram's cache file is written atomically on completion, so its presence means it is done
    pending = [tomogram_id for tomogram_id in picks_by_tomogram if not (cache_dir / f'{tomogram_id}.json').is_file()]
    if len(pending) < len(picks_by_tomogram):
        log.info(f'Resuming {name}: {len(picks_by_tomogram) - len(pending)} tomogram(s) loaded from checkpoint')
    for tomogram_id in pending:
        oriented = infer_orientations(
            picks_by_tomogram[tomogram_id],
            [manifest_by_id[tomogram_id]],
            vesicle_labels_by_tomogram=vesicle_labels_by_tomogram,
            max_surface_distance_angstrom=max_surface_distance_angstrom,
        ) if prior is OrientationPrior.NORMAL else picks_by_tomogram[tomogram_id]    # only membrane picks are oriented from the segmentation
        cache_tomogram_picks(cache_dir, tomogram_id, oriented)
    return load_cached_picks(cache_dir, list(picks_by_tomogram))

# _resolve_vesicle_labels: match --vesicle-labels-mrc (file or directory) to manifests by tomogram_id, warning per unmatched tomogram
def _resolve_vesicle_labels(vesicle_labels_mrc: Path | None, manifests: list[TomogramManifest]) -> dict[str, Path]:
    if vesicle_labels_mrc is None:
        return {}
    if vesicle_labels_mrc.is_file():
        if len(manifests) > 1:
            log.warning('--vesicle-labels-mrc is a single file but multiple tomograms are being picked; ignoring --vesicle-labels-mrc')
        return {}

    by_stem = {path.stem: path for path in glob_mrc(vesicle_labels_mrc)}
    resolved: dict[str, Path] = {}
    for manifest in manifests:
        match = by_stem.get(f'{manifest.tomogram_id}_labelled') or next(
            (path for stem, path in by_stem.items() if stem.startswith(manifest.tomogram_id)), None
        )
        if match is None:
            log.warning(f'No vesicle labels MRC matching {manifest.tomogram_id} in {vesicle_labels_mrc}, picking without vesicle labelling for it')
            continue
        resolved[manifest.tomogram_id] = match
    return resolved

# _VesicleSummaryJob: one tomogram's summary inputs
@dataclass(frozen=True)
class _VesicleSummaryJob:
    tomogram_id: str
    segmentation_path: Path
    labels_mrc: Path
    voxel_size_angstrom: float
    picks: list[RawPick]

# _summarise_one_tomogram: multiprocessing worker entry point
def _summarise_one_tomogram(job: _VesicleSummaryJob) -> list[dict]:
    segmentation = read_mrc(job.segmentation_path)
    labels = load_vesicle_labels(job.labels_mrc, segmentation.shape)
    vertices, _normals = extract_surface(segmentation)
    _v, faces, _n2, _val = measure.marching_cubes(segmentation.astype(np.float32), level=0.5)
    areas = vesicle_surface_area_angstrom2(vertices, faces, labels, job.voxel_size_angstrom, job.tomogram_id)
    return [summary.__dict__ for summary in summarise_vesicles(job.picks, areas, job.tomogram_id)]

# _write_vesicle_summary: per-vesicle pick count, area and density across all tomograms, written as vesicle_summary.json
def _write_vesicle_summary(
    all_reconciled: list[RawPick],
    manifests: list[TomogramManifest],
    vesicle_labels_mrc_by_tomogram: dict[str, Path],
    output_dir: Path,
    n_workers: int = 1,
) -> None:
    picks_by_tomogram: dict[str, list[RawPick]] = {}
    for pick in all_reconciled:
        picks_by_tomogram.setdefault(pick.tomogram_id, []).append(pick)

    jobs: list[_VesicleSummaryJob] = []
    for manifest in manifests:
        labels_mrc = vesicle_labels_mrc_by_tomogram.get(manifest.tomogram_id)
        picks = picks_by_tomogram.get(manifest.tomogram_id, [])
        if manifest.segmentation_path is None or labels_mrc is None or not any(p.vesicle_id for p in picks):
            continue
        jobs.append(_VesicleSummaryJob(manifest.tomogram_id, manifest.segmentation_path, labels_mrc, manifest.voxel_size_angstrom, picks))

    rows_by_tomogram: dict[str, list[dict]] = {}

    def _on_success(job: _VesicleSummaryJob, result: list[dict]) -> None:
        rows_by_tomogram[job.tomogram_id] = result

    def _on_error(job: _VesicleSummaryJob, exc: Exception) -> None:
        if isinstance(exc, StampValidationError):
            log.error(f'{job.tomogram_id}: vesicle summary failed ({exc})')
        else:
            raise exc

    run_parallel(jobs, _summarise_one_tomogram, max_workers=n_workers, label='vesicle-summary', on_success=_on_success, on_error=_on_error)

    rows = [row for job in jobs for row in rows_by_tomogram.get(job.tomogram_id, [])]
    if not rows:
        return
    summary_path = output_dir / 'vesicle_summary.json'
    summary_path.write_text(json.dumps(rows, indent=2))
    n_vesicles = len(rows)
    total_picks = sum(r['n_picks'] for r in rows)
    log.info(f'{n_vesicles} vesicle(s) across {len(manifests)} tomogram(s), {total_picks} picks ({total_picks / n_vesicles:.1f} picks/vesicle on average)')

# run_pick: pick command orchestration
def run_pick(
    picks_files: list[Path],
    segmentation_dir: Path | None,
    raw_tomogram_dir: Path,
    output_dir: Path,
    voxel_size_angstrom: float | None,
    consensus_rule: str,
    distance_threshold: float,
    half_set_seed: int,
    picker_names: list[str] | None = None,
    n_workers: int = 1,
    make_plots: bool = False,
    pick_plot_style: str = 'segmented',
    plot_format: str = 'tiff',
    pick_zstack_movie: bool = True,
    pick_plot_3d: bool = False,
    max_beam_angle_deviation: float | None = None,
    vesicle_labels_mrc: Path | None = None,
    max_surface_distance_angstrom: float = 300.0,
    keep_raw: bool = False,
    orientation_prior: OrientationPrior = OrientationPrior.NORMAL,
) -> None:
    if orientation_prior is OrientationPrior.NORMAL and segmentation_dir is None:
        raise StampPipelineError('--seg-dir is required for the normal orientation prior, orientations are inferred from the membrane segmentation')
    names = picker_names or [path.stem for path in picks_files]
    if len(names) != len(picks_files):
        raise StampPipelineError(f'{len(picks_files)} picks file(s) but {len(names)} picker name(s)')
    if len(set(names)) != len(names):
        raise StampPipelineError(f'picker names must be unique, got {names!r}; use --name to disambiguate files with the same stem')

    # Load manifests to check for matching files
    manifests = load_tomogram_manifests(segmentation_dir, raw_tomogram_dir, voxel_size_angstrom)
    if not manifests:
        raise StampPipelineError(f'No .mrc files in {segmentation_dir or raw_tomogram_dir} with a matching stem in {raw_tomogram_dir}')
    log.info(f'Matched {len(manifests)} tomograms/segmentations')
    resolved_voxel_size_angstrom = manifests[0].voxel_size_angstrom
    log.debug(f'Resolved voxel size: {resolved_voxel_size_angstrom} Å')

    output_dir.mkdir(parents=True, exist_ok=True)
    vesicle_labels_mrc_by_tomogram = _resolve_vesicle_labels(vesicle_labels_mrc, manifests)

    # everything that changes imported picks; consensus, seed, beam filter, workers and plots are left out so they can change between runs
    fingerprint = {
        'picks': {name: file_signature(path, content=True) for name, path in zip(names, picks_files)},
        'segmentation': signatures({m.tomogram_id: m.segmentation_path for m in manifests if m.segmentation_path}),
        'orientation_prior': orientation_prior.value,
        'raw': signatures({m.tomogram_id: m.raw_tomogram_path for m in manifests}),
        'vesicle_labels': signatures(vesicle_labels_mrc_by_tomogram),
        'voxel_size_angstrom': resolved_voxel_size_angstrom,
        'max_surface_distance_angstrom': max_surface_distance_angstrom,
    }
    if sync_checkpoint(output_dir / 'raw', fingerprint):
        log.info('Found matching partial output, resuming')

    picks_by_picker: dict[str, list[RawPick]] = {}
    for name, path in zip(names, picks_files):
        log.progress(f'Importing {name} from {path.name}...')
        picks_by_picker[name] = _import_picker(path, name, manifests, vesicle_labels_mrc_by_tomogram, max_surface_distance_angstrom, output_dir / 'raw' / name, orientation_prior)
        log.info(f'{name}: {len(picks_by_picker[name])} raw picks')

    log.progress(f'Reconciling picks...')
    all_reconciled = reconcile_picks_for_tomograms(
        picks_by_picker=picks_by_picker,
        consensus_rule=consensus_rule,  # type: ignore[arg-type]
        distance_threshold=distance_threshold,
        tomogram_ids=[manifest.tomogram_id for manifest in manifests],
        n_workers=n_workers,
    )
    if not all_reconciled:
        raise StampPipelineError('No particles survived reconciliation. With --consensus-rule intersection, every picks file must have a pick within --distance-threshold voxels of each particle.')

    log.progress(f'Building particle set and determining beam angle distribution')
    particle_set = build_particle_set(
        reconciled_picks=all_reconciled,
        consensus_rule=consensus_rule,  # type: ignore[arg-type]
        contributing_pickers=names,
        half_set_seed=half_set_seed,
        orientation_prior=orientation_prior,
    )
    if max_beam_angle_deviation is not None:
        before = len(particle_set.particles)
        particle_set.particles = [
            p for p in particle_set.particles
            if p.beam_angle_deviation_degrees is None or p.beam_angle_deviation_degrees <= max_beam_angle_deviation
        ]
        dropped = before - len(particle_set.particles)
        log.info(f'--max-beam-angle-deviation {max_beam_angle_deviation}: dropped {dropped} of {before} particles')
        if not particle_set.particles:
            raise StampPipelineError('No particles survived the beam angle deviation filter. Try raising --max-beam-angle-deviation.')
    if orientation_prior is OrientationPrior.NORMAL:
        report_beam_angle_distribution(particle_set.particles)
    particle_set_path = output_dir / 'particle_set.json'
    particle_set_path.write_text(particle_set.model_dump_json(indent=2))
    log.info(f'Wrote {len(particle_set.particles)} consensus particles to {particle_set_path}')

    log.progress('Computing vesicle summary')
    _write_vesicle_summary(all_reconciled, manifests, vesicle_labels_mrc_by_tomogram, output_dir, n_workers=n_workers)
    log.progress(f'Writing parameters...')
    write_sidecar(
        output_dir,
        stage='pick',
        tool='+'.join(names),
        tool_version=None,
        parameters={
            'pickers': names,
            'picks_files': [str(path) for path in picks_files],
            'consensus_rule': consensus_rule,
            'distance_threshold': distance_threshold,
            'half_set_seed': half_set_seed,
            'orientation_prior': orientation_prior.value,
            'voxel_size_angstrom': resolved_voxel_size_angstrom,
            'max_surface_distance_angstrom': max_surface_distance_angstrom,
            'vesicle_labels_mrc': {tomogram_id: str(path) for tomogram_id, path in vesicle_labels_mrc_by_tomogram.items()},
        },
        inputs=[(f'picks:{name}', path) for name, path in zip(names, picks_files)]
        + [(f'segmentation:{m.tomogram_id}', m.segmentation_path) for m in manifests if m.segmentation_path]
        + [(f'raw_tomogram:{m.tomogram_id}', m.raw_tomogram_path) for m in manifests],
    )

    log.progress(f'Rendering plots...')
    if make_plots:
        segmentation_paths = {m.tomogram_id: m.segmentation_path for m in manifests if m.segmentation_path}
        raw_tomogram_paths = {m.tomogram_id: m.raw_tomogram_path for m in manifests}
        plot_positions(particle_set.particles, output_dir, pick_plot_style, plot_format, segmentation_paths, raw_tomogram_paths, zstack_movie=pick_zstack_movie, plot_3d_view=pick_plot_3d, max_workers=n_workers)

    if not keep_raw:
        raw_dir = output_dir / 'raw'
        if raw_dir.is_dir():
            archive_path = archive_and_remove_directory(raw_dir)
            log.info(f'Archived imported pick cache to {archive_path}')

# build_pick_commands: the ToolCommand `stamp pick` would run for the real track, without running it
def build_pick_commands(config, output_dir: Path) -> list[ToolCommand]:
    target = stage_dir(output_dir, 'real', 'pick')
    argv = [
        'stamp', 'pick', *[str(path) for path in config.stage.pick.picks_files],
        '--orientation-prior', config.run.orientation_prior.value,
        '--raw-dir', str(config.run.raw_tomogram_dir),
        '--out-dir', str(output_dir),
        '--voxel-size-a', str(config.run.voxel_size_angstrom),
        '--consensus-rule', config.stage.pick.consensus_rule,
        '--distance-threshold', str(config.stage.pick.distance_threshold),
        '--half-set-seed', str(config.stage.pick.half_set_seed),
        '--max-surface-distance-a', str(config.stage.pick.max_surface_distance_angstrom),
        '--n-processes', str(config.stage.pick.n_workers),
    ]
    if config.run.segmentation_dir is not None:
        argv += ['--seg-dir', str(config.run.segmentation_dir)]
    for name in config.stage.pick.picker_names or []:
        argv += ['--name', name]
    if config.plots.enabled:
        argv.append('--plots')
        argv += ['--pick-plot-style', config.plots.pick_style, '--plot-format', config.plots.format]
    if config.plots.pick_zstack_movie:
        argv.append('--pick-zstack-movie')
    if config.plots.pick_plot_3d:
        argv.append('--pick-plot-3d')
    if config.stage.pick.keep_raw:
        argv.append('--keep-raw')
    return [ToolCommand(tool='pick', argv=argv, working_directory=target, output_paths=[target / 'particle_set.json'])]
