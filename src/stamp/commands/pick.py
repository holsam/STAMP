'''
STAMP: consensus picking logic
'''

# Import external dependencies
import json, mrcfile, numpy as np
from dataclasses import dataclass
from pathlib import Path
from skimage import measure

# Import internal STAMP objects
from stamp.adapters.base import AdapterInputs, ToolAdapter
from stamp.adapters.mock import get_mock_adapter
from stamp.adapters.native import NativePickerAdapter
from stamp.backends.base import Runner, ToolCommand
from stamp.backends.factory import check_backend_supports, select_runner
from stamp.picking.consensus import build_particle_set, reconcile_picks_for_tomograms
from stamp.picking.geometry import extract_surface
from stamp.picking.native import PICKER_NAME as NATIVE_PICKER_NAME
from stamp.picking.vesicles import load_vesicle_labels, summarise_vesicles, vesicle_surface_area_angstrom2
from stamp.run.state import stage_dir
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.picks import RawPick
from stamp.utils.errors import StampPipelineError, StampValidationError
from stamp.utils.io import archive_and_remove_directory, match_by_stem, resolve_directory_voxel_size_angstrom, write_sidecar
from stamp.utils.log import log
from stamp.utils.parallel import run_parallel
from stamp.utils.plotting.picks import plot_positions
from stamp.utils.reporting import report_beam_angle_distribution

# REAL_ADAPTERS: dictionary containing all implemented pickers
REAL_ADAPTERS: dict[str, ToolAdapter] = {
    NATIVE_PICKER_NAME: NativePickerAdapter(),
}

# _execute: build a command, run it, and parse the resulting RawPicks
def _execute(adapter: ToolAdapter, inputs: AdapterInputs, runner: Runner) -> list[RawPick]:
    command = adapter.build_command(inputs)
    log.debug(f'{adapter.name}: dispatching {" ".join(command.argv)}')
    result = runner.run(command)
    if not result.succeeded:
        raise StampPipelineError(f'{adapter.name} failed (exit {result.exit_code}): {result.stderr}')
    output = adapter.parse_output(result)
    return [RawPick(**raw) for raw in output.parsed.get('picks', [])]

# _select_adapter: return the adapter for a picker name, guarding it against the chosen backend
def _select_adapter(picker_name: str, backend: str) -> ToolAdapter:
    if backend == 'mock':
        return get_mock_adapter(picker_name)

    adapter = REAL_ADAPTERS.get(picker_name)
    if adapter is None:
        raise StampPipelineError(f'Unknown picker {picker_name!r}')
    check_backend_supports(adapter, backend)
    return adapter

# _load_manifests: match segmentations to raw tomograms by filename stem
def _load_manifests(
    segmentation_dir: Path,
    raw_tomogram_dir: Path,
    voxel_size_angstrom: float | None,
) -> list[TomogramManifest]:
    '''Unmatched segmentations are skipped with a warning rather than failing the run. Voxel size is read from each raw tomogram's MRC header when not given explicitly.'''
    segmentation_paths = sorted(segmentation_dir.glob('*.mrc'))
    raw_paths = sorted(raw_tomogram_dir.glob('*.mrc'))
    matched, unmatched = match_by_stem(segmentation_paths, raw_paths)
    for segmentation_path in unmatched:
        log.warning(f'No raw tomogram matching {segmentation_path.stem}, skipping')
    manifests: list[TomogramManifest] = []
    if voxel_size_angstrom is None:
        resolved_voxel_size = resolve_directory_voxel_size_angstrom(list(matched.values()))
        if resolved_voxel_size is None:
            raise StampPipelineError(f'No voxel size in any raw tomogram header under {raw_tomogram_dir} and --voxel-size-a not given')
    for segmentation_path, raw_path in matched.items():
        manifests.append(
            TomogramManifest(
                tomogram_id=segmentation_path.stem,
                segmentation_path=segmentation_path,
                raw_tomogram_path=raw_path,
                voxel_size_angstrom=voxel_size_angstrom or resolved_voxel_size,
            )
        )
    return manifests

# _run_picker: dispatch to the in-process, batched, or per-tomogram path
def _run_picker(
    adapter: ToolAdapter,
    manifests: list[TomogramManifest],
    picker_output_dir: Path,
    parameters: dict,
    runner: Runner,
    backend: str,
) -> list[RawPick]:
    if backend != 'mock' and getattr(adapter, 'runs_in_process', False):
        inputs = AdapterInputs(
            input_paths=[m.segmentation_path for m in manifests],
            raw_tomogram_paths=[m.raw_tomogram_path for m in manifests],
            tomogram_ids=[m.tomogram_id for m in manifests],
            output_directory=picker_output_dir,
            parameters=parameters,
        )
        return adapter.run_in_process(inputs)  # type: ignore[attr-defined]

    if backend == 'mock' or adapter.batches_natively:
        inputs = AdapterInputs(
            input_paths=[m.segmentation_path for m in manifests],
            raw_tomogram_paths=[m.raw_tomogram_path for m in manifests],
            tomogram_ids=[m.tomogram_id for m in manifests],
            output_directory=picker_output_dir,
            parameters=parameters,
        )
        return _execute(adapter, inputs, runner)

    collected: list[RawPick] = []
    for manifest in manifests:
        inputs = AdapterInputs(
            input_paths=[manifest.segmentation_path],
            raw_tomogram_paths=[manifest.raw_tomogram_path],
            tomogram_ids=[manifest.tomogram_id],
            output_directory=picker_output_dir / manifest.tomogram_id,
            parameters=parameters,
        )
        collected.extend(_execute(adapter, inputs, runner))
    return collected

# _resolve_vesicle_labels: match --vesicle-labels-mrc (file or directory) to manifests by tomogram_id, warning per unmatched tomogram
def _resolve_vesicle_labels(vesicle_labels_mrc: Path | None, manifests: list[TomogramManifest]) -> dict[str, Path]:
    if vesicle_labels_mrc is None:
        return {}
    if vesicle_labels_mrc.is_file():
        if len(manifests) > 1:
            log.warning('--vesicle-labels-mrc is a single file but multiple tomograms are being picked; ignoring --vesicle-labels-mrc')
        return {}

    by_stem = {path.stem: path for path in sorted(vesicle_labels_mrc.glob('*.mrc'))}
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
    with mrcfile.open(str(job.segmentation_path), permissive=True) as mrc:
        segmentation = np.asarray(mrc.data)
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
        if labels_mrc is None or not any(p.vesicle_id for p in picks):
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
    picker_names,
    segmentation_dir,
    raw_tomogram_dir,
    output_dir,
    voxel_size_angstrom,
    extra_params,
    consensus_rule,
    distance_threshold,
    half_set_seed,
    backend,
    n_workers: int = 1,
    make_plots: bool = True,
    pick_plot_style: str = 'both',
    plot_format: str = 'tiff',
    pick_zstack_movie: bool = True,
    max_beam_angle_deviation: float | None = None,
    vesicle_labels_mrc: Path | None = None,
    normalise_per_vesicle: bool = False,
    keep_raw: bool = False,
) -> None:
    # Load manifests to check for matching files
    manifests = _load_manifests(segmentation_dir, raw_tomogram_dir, voxel_size_angstrom)
    if not manifests:
        raise StampPipelineError(f'No .mrc files in {segmentation_dir} with a matching stem in {raw_tomogram_dir}')
    log.info(f'Matched {len(manifests)} tomograms/segmentations')
    resolved_voxel_size_angstrom = manifests[0].voxel_size_angstrom

    runner = select_runner(backend)
    output_dir.mkdir(parents=True, exist_ok=True)

    vesicle_labels_mrc_by_tomogram = _resolve_vesicle_labels(vesicle_labels_mrc, manifests)
    if normalise_per_vesicle and not vesicle_labels_mrc_by_tomogram:
        raise StampPipelineError('--normalise-per-vesicle requires --vesicle-labels-mrc to resolve to at least one tomogram')

    picks_by_picker: dict[str, list[RawPick]] = {}
    for picker_name in picker_names:
        adapter = _select_adapter(picker_name, backend)
        picker_output_dir = output_dir / 'raw' / picker_name
        parameters = {
            'voxel_size_angstrom': resolved_voxel_size_angstrom,
            **extra_params.get(picker_name, {}),
        }
        if picker_name == NATIVE_PICKER_NAME:
            parameters.setdefault('n_workers', n_workers)
            if vesicle_labels_mrc_by_tomogram:
                parameters['vesicle_labels_mrc_by_tomogram'] = {tomogram_id: str(path) for tomogram_id, path in vesicle_labels_mrc_by_tomogram.items()}
                parameters.setdefault('normalise_per_vesicle', normalise_per_vesicle)

        log.progress(f'Running {picker_name}...')
        picks_by_picker[picker_name] = _run_picker(adapter, manifests, picker_output_dir, parameters, runner, backend)
        log.info(f'{picker_name}: {len(picks_by_picker[picker_name])} raw picks')

    log.progress(f'Reconciling picks...')
    all_reconciled = reconcile_picks_for_tomograms(
        picks_by_picker=picks_by_picker,
        consensus_rule=consensus_rule,  # type: ignore[arg-type]
        distance_threshold=distance_threshold,
        tomogram_ids=[manifest.tomogram_id for manifest in manifests],
        n_workers=n_workers,
    )
    if not all_reconciled:
        raise StampPipelineError('No particles survived reconciliation. If using stamp-native, try lowering n_mad or check density_sign matches your tomograms (-1 for conventional dark-protein contrast).')

    log.progress(f'Building particle set and determining beam angle distribution')
    particle_set = build_particle_set(
        reconciled_picks=all_reconciled,
        consensus_rule=consensus_rule,  # type: ignore[arg-type]
        contributing_pickers=picker_names,
        half_set_seed=half_set_seed,
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
    report_beam_angle_distribution(particle_set.particles)
    particle_set_path = output_dir / 'particle_set.json'
    particle_set_path.write_text(particle_set.model_dump_json(indent=2))
    log.info(f'Wrote {len(particle_set.particles)} consensus particles to {particle_set_path}')

    _write_vesicle_summary(all_reconciled, manifests, vesicle_labels_mrc_by_tomogram, output_dir, n_workers=n_workers)
    log.progress(f'Writing parameters...')
    write_sidecar(
        output_dir,
        stage='pick',
        tool='+'.join(picker_names),
        tool_version=None,
        parameters={
            'pickers': picker_names,
            'consensus_rule': consensus_rule,
            'distance_threshold': distance_threshold,
            'half_set_seed': half_set_seed,
            'voxel_size_angstrom': resolved_voxel_size_angstrom,
            'backend': backend,
            'picker_params': extra_params,
            'vesicle_labels_mrc': {tomogram_id: str(path) for tomogram_id, path in vesicle_labels_mrc_by_tomogram.items()},
        },
        inputs=[(f'segmentation:{m.tomogram_id}', m.segmentation_path) for m in manifests] + [(f'raw_tomogram:{m.tomogram_id}', m.raw_tomogram_path) for m in manifests],
    )

    log.progress(f'Rendering plots...')
    if make_plots:
        segmentation_paths = {m.tomogram_id: m.segmentation_path for m in manifests}
        raw_tomogram_paths = {m.tomogram_id: m.raw_tomogram_path for m in manifests}
        plot_positions(particle_set.particles, output_dir, pick_plot_style, plot_format, segmentation_paths, raw_tomogram_paths, zstack_movie=pick_zstack_movie, max_workers=n_workers)

    if not keep_raw:
        raw_dir = output_dir / 'raw'
        if raw_dir.is_dir():
            archive_path = archive_and_remove_directory(raw_dir)
            log.info(f'Archived raw picker output to {archive_path}')

# build_pick_commands: the ToolCommand `stamp pick` would run for the real track, without running it
def build_pick_commands(config, output_dir: Path) -> list[ToolCommand]:
    target = stage_dir(output_dir, 'real', 'pick')
    argv = [
        'stamp', 'pick', ','.join(config.stage.pick.pickers),
        '--seg-dir', str(config.run.segmentation_dir),
        '--raw-dir', str(config.run.raw_tomogram_dir),
        '--out-dir', str(target),
        '--voxel-size-a', str(config.run.voxel_size_angstrom),
        '--consensus-rule', config.stage.pick.consensus_rule,
        '--distance-threshold', str(config.stage.pick.distance_threshold),
        '--half-set-seed', str(config.stage.pick.half_set_seed),
        '--backend', 'local',
    ]
    argv.append('--plots' if config.plots.enabled else '--no-plots')
    argv += ['--pick-plot-style', config.plots.pick_style, '--plot-format', config.plots.format]
    if config.plots.pick_zstack_movie:
        argv.append('--pick-zstack-movie')
    argv.append('--keep-raw' if config.stage.pick.keep_raw else '--no-keep-raw')
    return [ToolCommand(tool='pick', argv=argv, working_directory=target, output_paths=[target / 'particle_set.json'])]