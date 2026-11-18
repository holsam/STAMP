'''
STAMP: stream logic
'''

# Import external dependencies
import json, shutil
from dataclasses import asdict
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.extract import extract_particle_set
from stamp.commands.classify import build_group_settings, classify_particle_set
from stamp.commands.pick import pick_tomogram
from stamp.decoy.validate import assert_comparable
from stamp.utils.halfset import assign_half_sets
from stamp.utils.provenance import write_sidecar
from stamp.decoy.generate import (
    METHOD_RANDOM_VOLUME,
    METHOD_SYNTHETIC_NOISE,
    DecoyGeometry,
    generate_random_volume_decoys,
    generate_shifted_decoys,
    merge_decoy_sets,
    resolve_methods,
)
from stamp.picking.vesicles import resolve_vesicle_labels, write_vesicle_summary
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.particles import HalfSet, OrientationPrior, Particle, ParticleSet, load_particle_set, particle_ids
from stamp.schemas.picks import RawPick
from stamp.stream.state import StreamDir, StreamSettings, load_or_init_settings
from stamp.utils.checkpoint import atomic_write_json, file_signature
from stamp.utils.errors import StampPipelineError, StampValidationError
from stamp.utils.log import log
from stamp.utils.mrc import new_mrc, open_mrc
from stamp.utils.plotting.picks import plot_positions

# _particles_from_picks: real particles for one tomogram
def _particles_from_picks(picks: list[RawPick]) -> list[Particle]:
    ids = particle_ids([pick.tomogram_id for pick in picks], 'p')
    return [
        Particle(
            particle_id=particle_id,
            tomogram_id=pick.tomogram_id,
            position=pick.position,
            orientation=pick.orientation,
            prior=pick.prior,
            source_picker=pick.source_picker,
            confidence=pick.confidence,
            half_set=HalfSet.A,
            beam_angle_deviation_degrees=pick.beam_angle_deviation_degrees,
            vesicle_id=pick.vesicle_id,
        )
        for particle_id, pick in zip(ids, picks)
    ]

# _set_payload: a particle set as JSON, written directly so an empty tomogram can be recorded
def _set_payload(particles: list[Particle], consensus_rule: str, contributing_pickers: list[str]) -> dict:
    return {
        'particles': [particle.model_dump(mode='json') for particle in particles],
        'consensus_rule': consensus_rule,
        'contributing_pickers': contributing_pickers,
    }

# _export_particles: write each kept real particle as one raw (not membrane-subtracted) oriented box MRC
def _export_particles(
    particles: list[Particle],
    tomogram_id: str,
    tomogram_path: Path,
    stream: StreamDir,
    box_voxels: int,
    voxel_size_angstrom: float,
    n_workers: int,
) -> int:
    export_dir = stream.particle_dir(tomogram_id)
    scratch = export_dir / '.scratch'   # extraction cache, removed once the MRCs are written
    shutil.rmtree(scratch, ignore_errors=True)
    by_prior: dict[OrientationPrior, list[Particle]] = {}
    for particle in particles:
        by_prior.setdefault(particle.prior, []).append(particle)
    n_written = 0
    for prior, group in by_prior.items():
        subvols, kept, _ = extract_particle_set(group, {tomogram_id: str(tomogram_path)}, box_voxels, segmentation_paths={}, n_workers=n_workers, cache_dir=scratch / prior.value, prior=prior)
        export_dir.mkdir(parents=True, exist_ok=True)
        for row, particle in enumerate(kept):   # one tomogram, so kept[row] is the row of subvols[row]
            new_mrc(export_dir / f'{particle.particle_id}.mrc', subvols[row], voxel_size=voxel_size_angstrom)
            n_written += 1
    shutil.rmtree(scratch, ignore_errors=True)
    return n_written

# _decoys_for_tomogram: decoys for one tomogram's real particles, by the methods run_decoy would use, with no generator-level cache
def _decoys_for_tomogram(real: ParticleSet, manifest: TomogramManifest, settings: StreamSettings) -> ParticleSet | None:
    config = DecoyGeometry(voxel_size_angstrom=settings.voxel_size_angstrom, density_sign=settings.density_sign)
    present = {particle.prior for particle in real.particles}
    decoy_sets: list[ParticleSet] = []
    try:
        for method in resolve_methods(settings.decoy_method, present):
            if method == METHOD_RANDOM_VOLUME:
                decoys = generate_random_volume_decoys(
                    real_particle_set=real, manifests=[manifest], config=config,
                    min_distance_from_picks_angstrom=settings.min_distance_from_picks_angstrom,
                    border_angstrom=settings.min_border_angstrom, seed=settings.seed,
                )
            elif OrientationPrior.NORMAL in present:
                decoys = generate_shifted_decoys(
                    real_particle_set=real, manifests=[manifest], config=config,
                    min_shift_angstrom=settings.min_shift_angstrom, max_shift_angstrom=settings.max_shift_angstrom,
                    min_distance_from_surface_angstrom=settings.min_distance_from_real_angstrom,
                    min_pick_distance=settings.min_distance_from_picks_angstrom, seed=settings.seed,
                )
            else:
                continue
            if decoys is not None:
                decoy_sets.append(decoys)
    except StampValidationError as exc:
        raise StampPipelineError(str(exc)) from exc
    return merge_decoy_sets(decoy_sets, settings.seed)

# _extract_set: subvolume extraction for one set ('real' or 'decoy'), one prior at a time, into the stream's cache layout
def _extract_set(
    particles: list[Particle],
    kind: str,
    stream: StreamDir,
    tomogram_path: Path,
    box_voxels: int,
    segmentation_paths: dict[str, str],
    n_workers: int,
) -> tuple[dict[str, int], int]:
    by_prior: dict[OrientationPrior, list[Particle]] = {}
    for particle in particles:
        by_prior.setdefault(particle.prior, []).append(particle)
    n_kept_by_prior: dict[str, int] = {}
    n_skipped = 0
    for prior, group in by_prior.items():
        _, kept, skipped = extract_particle_set(
            group, {group[0].tomogram_id: str(tomogram_path)}, box_voxels,
            segmentation_paths=segmentation_paths, n_workers=n_workers,
            cache_dir=stream.subvolume_dir(kind, prior.value), prior=prior,
        )
        n_kept_by_prior[prior.value] = len(kept)
        n_skipped += len(skipped)
    return n_kept_by_prior, n_skipped

# run_stream_append: pick, decoy and extract one tomogram into the stream directory, writing done.json last so an interrupted append restarts that tomogram only
def run_stream_append(
    *,
    stream_dir: Path,
    picks_files: list[Path],
    picker_names: list[str],
    tomogram_path: Path,
    segmentation_path: Path | None,
    vesicle_labels_mrc: Path | None,
    requested: StreamSettings,
    n_workers: int = 1,
    make_plots: bool = False,
    plot_format: str = 'tiff',
    pick_plot_style: str = 'segmented',
    pick_zstack_movie: bool = True,
    pick_plot_3d: bool = False,
    export_particles: bool = False,
) -> None:
    if requested.decoy_method == METHOD_SYNTHETIC_NOISE:
        raise StampPipelineError('stream append does not support --decoy-method synthetic-noise (it makes its own volumes, not tomogram decoys), use `stamp decoy --method synthetic-noise` instead')
    if requested.decoy_method not in ('auto', 'shifted', 'random-volume'):
        raise StampPipelineError(f'unknown --decoy-method {requested.decoy_method!r}')

    stream = StreamDir(stream_dir)
    settings = load_or_init_settings(stream, requested)
    tomogram_id = tomogram_path.stem
    if stream.is_done(tomogram_id):
        log.info(f'{tomogram_id}: already appended, skipping')
        return

    with open_mrc(tomogram_path, header_only=True) as mrc:
        shape_xyz = [int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)]
        header_voxel = float(mrc.voxel_size.x)
    if header_voxel > 0 and abs(header_voxel - settings.voxel_size_angstrom) > 1e-3:
        log.warning(f'{tomogram_id}: MRC header voxel size {header_voxel} Å differs from --voxel-size-a {settings.voxel_size_angstrom} Å')

    manifest = TomogramManifest(
        tomogram_id=tomogram_id,
        segmentation_path=segmentation_path,
        raw_tomogram_path=tomogram_path,
        voxel_size_angstrom=settings.voxel_size_angstrom,
    )
    picks = pick_tomogram(
        picks_files, picker_names, manifest, vesicle_labels_mrc,
        consensus_rule=settings.consensus_rule, distance_threshold=settings.distance_threshold,
        max_surface_distance_angstrom=settings.max_surface_distance_angstrom,
        max_beam_angle_deviation=settings.max_beam_angle_deviation,
        orientation_prior=OrientationPrior(settings.orientation_prior),
    )
    if not picks:
        log.warning(f'{tomogram_id}: no particles after picking, recording an empty tomogram')

    real_particles = _particles_from_picks(picks)
    decoy_set = _decoys_for_tomogram(ParticleSet(particles=real_particles, consensus_rule=settings.consensus_rule, contributing_pickers=picker_names), manifest, settings) if real_particles else None
    decoy_particles = decoy_set.particles if decoy_set is not None else []

    segmentation_paths = {tomogram_id: str(segmentation_path)} if segmentation_path is not None else {}
    n_kept_real, n_skipped_real = _extract_set(real_particles, 'real', stream, tomogram_path, settings.box_voxels, segmentation_paths, n_workers)
    n_kept_decoy, n_skipped_decoy = _extract_set(decoy_particles, 'decoy', stream, tomogram_path, settings.box_voxels, segmentation_paths, n_workers)

    if export_particles and real_particles:
        n_exported = _export_particles(real_particles, tomogram_id, tomogram_path, stream, settings.box_voxels, settings.voxel_size_angstrom, n_workers)
        log.info(f'{tomogram_id}: exported {n_exported} particle box(es)')

    tomogram_dir = stream.tomogram_dir(tomogram_id)
    tomogram_dir.mkdir(parents=True, exist_ok=True)
    labels = resolve_vesicle_labels(vesicle_labels_mrc, [manifest])
    write_vesicle_summary(picks, [manifest], labels, tomogram_dir, n_workers=n_workers)

    # plots are written while the tomogram is still on disk
    if make_plots:
        segmentation_map = {tomogram_id: segmentation_path} if segmentation_path is not None else {}
        raw_map = {tomogram_id: tomogram_path}
        if real_particles:
            plot_positions(real_particles, tomogram_dir, pick_plot_style, plot_format, segmentation_map, raw_map, zstack_movie=pick_zstack_movie, plot_3d_view=pick_plot_3d, max_workers=n_workers)
        if decoy_particles:
            plot_positions(decoy_particles, tomogram_dir / 'decoy', pick_plot_style, plot_format, segmentation_map, raw_map, zstack_movie=pick_zstack_movie, plot_3d_view=pick_plot_3d, max_workers=n_workers)

    atomic_write_json(tomogram_dir / 'particles.json', _set_payload(real_particles, settings.consensus_rule, picker_names))
    atomic_write_json(tomogram_dir / 'decoy_particles.json', _set_payload(decoy_particles, 'union', decoy_set.contributing_pickers if decoy_set is not None else []))

    atomic_write_json(tomogram_dir / 'done.json', {
        'tomogram_id': tomogram_id,
        'n_particles': len(real_particles),
        'n_decoys': len(decoy_particles),
        'n_kept_by_prior': {'real': n_kept_real, 'decoy': n_kept_decoy},
        'n_skipped': {'real': n_skipped_real, 'decoy': n_skipped_decoy},
        'shape_xyz': shape_xyz,
    })
    log.info(f'{tomogram_id}: appended {len(real_particles)} particles and {len(decoy_particles)} decoys')

# _load_set_payload: particles and contributing pickers from one tomogram's particle set JSON
def _load_set_payload(path: Path) -> tuple[list[Particle], list[str]]:
    payload = json.loads(path.read_text())
    return [Particle.model_validate(particle) for particle in payload['particles']], payload['contributing_pickers']

# _with_half_sets: assign half-sets once over the whole stream, so whole tomograms land in one half
def _with_half_sets(particles: list[Particle], seed: int) -> list[Particle]:
    group_of = {particle.particle_id: particle.tomogram_id for particle in particles}
    half_by_id = assign_half_sets([particle.particle_id for particle in particles], seed=seed, group_of=group_of)
    return [particle.model_copy(update={'half_set': half_by_id[particle.particle_id]}) for particle in particles]

# run_stream_finalise: merge appended tomograms into one real and one decoy particle set with global half-sets, check comparability, then mark the stream finalised last
def run_stream_finalise(*, stream_dir: Path) -> None:
    stream = StreamDir(stream_dir)
    if not stream.settings_path.is_file():
        raise StampValidationError(f'{stream_dir} is not a stream directory (no stream.json)')
    settings = StreamSettings.model_validate(json.loads(stream.settings_path.read_text()))
    done = stream.done_tomograms()
    if not done:
        raise StampPipelineError(f'{stream_dir} has no appended tomograms to finalise')
    log.info(f'Finalising {len(done)} appended tomogram(s)')

    real: list[Particle] = []
    decoy: list[Particle] = []
    decoy_pickers: set[str] = set()
    tomogram_inputs: list[tuple[str, Path]] = []
    for tomogram_id in done:
        tomogram_dir = stream.tomogram_dir(tomogram_id)
        particles, _ = _load_set_payload(tomogram_dir / 'particles.json')
        decoys, pickers = _load_set_payload(tomogram_dir / 'decoy_particles.json')
        real.extend(particles)
        decoy.extend(decoys)
        decoy_pickers.update(pickers)
        tomogram_inputs.append((f'particles:{tomogram_id}', tomogram_dir / 'particles.json'))
    if not real:
        raise StampPipelineError('no real particles in the appended tomograms, nothing to finalise')
    if not decoy:
        raise StampPipelineError('no decoy particles in the appended tomograms, nothing to finalise')
    if len({particle.tomogram_id for particle in real}) < 2:
        log.warning('One tomogram only; half-sets are not independent (all particles in half A)')

    real_set = ParticleSet(particles=_with_half_sets(real, settings.half_set_seed), consensus_rule=settings.consensus_rule, contributing_pickers=settings.picker_names)
    decoy_set = ParticleSet(particles=_with_half_sets(decoy, settings.seed), consensus_rule='union', contributing_pickers=sorted(decoy_pickers))
    assert_comparable(real_set, decoy_set)

    atomic_write_json(stream.root / 'particle_set.json', real_set.model_dump(mode='json'))
    atomic_write_json(stream.root / 'decoy_particle_set.json', decoy_set.model_dump(mode='json'))

    vesicle_rows: list[dict] = []
    for tomogram_id in done:
        summary = stream.tomogram_dir(tomogram_id) / 'vesicle_summary.json'
        if summary.is_file():
            vesicle_rows.extend(json.loads(summary.read_text()))
    if vesicle_rows:
        atomic_write_json(stream.root / 'vesicle_summary.json', vesicle_rows)

    write_sidecar(
        stream.root,
        stage='stream-finalise',
        tool='stamp-stream',
        tool_version=None,
        parameters={
            'n_tomograms': len(done),
            'n_particles': len(real),
            'n_decoys': len(decoy),
            'consensus_rule': settings.consensus_rule,
            'half_set_seed': settings.half_set_seed,
            'seed': settings.seed,
        },
        inputs=tomogram_inputs,
    )

    # last, so a failure anywhere above leaves the stream open to retry
    settings.finalised = True
    atomic_write_json(stream.settings_path, settings.model_dump())
    log.info(f'Wrote {len(real)} particles and {len(decoy)} decoys to {stream.root}')

# _membrane_subtracted: True when any appended tomogram's subvolumes of this kind were membrane-subtracted at append time (recorded in the markers)
def _membrane_subtracted(stream: StreamDir, kind: str) -> bool:
    return any(json.loads(marker.read_text()).get('subtracted') for marker in (stream.root / 'subvolumes' / kind).glob('*/*.json'))

# run_stream_classify: called by `stamp classify` in stream mode, classifies finalised real and decoy sets from append-time subvolume cache, with no tomogram on disk
def run_stream_classify(
    *,
    stream_dir: Path,
    classify_options: dict,
    make_plots: bool,
    plot_format: str,
    keep_raw: bool,
) -> None:
    stream = StreamDir(stream_dir)
    if not stream.settings_path.is_file():
        raise StampValidationError(f'{stream_dir} is not a stream directory (no stream.json)')
    settings = StreamSettings.model_validate(json.loads(stream.settings_path.read_text()))
    if not settings.finalised:
        raise StampPipelineError(f'{stream_dir} is not finalised, run `stamp stream finalise` first')
    group_settings = build_group_settings(settings.box_voxels, settings.voxel_size_angstrom, **classify_options)
    box_angstrom = settings.box_voxels * settings.voxel_size_angstrom

    for kind, filename in (('real', 'particle_set.json'), ('decoy', 'decoy_particle_set.json')):
        path = stream_dir / filename
        if not path.is_file():
            raise StampValidationError(f'{path} is missing, run `stamp stream finalise` first')
        particle_set = load_particle_set(path)
        # everything that changes results, from the finalised sets and the stream settings
        fingerprint = {
            'particles': file_signature(path, content=True),
            'stream': settings.model_dump(),
            **{key: value for key, value in asdict(group_settings).items() if key != 'n_workers'},
        }
        classify_particle_set(
            particle_set, group_settings, stream.root / 'classify' / kind,
            tomogram_paths={}, segmentation_paths={},
            fingerprint=fingerprint,
            sidecar_inputs=[('particle_set', path)],
            extra_parameters={'box_angstrom': box_angstrom},
            keep_raw=keep_raw, make_plots=make_plots, plot_format=plot_format,
            cache_only=True,
            subvolume_cache_dir=lambda prior, kind=kind: stream.subvolume_dir(kind, prior.value),
            membrane_subtracted=_membrane_subtracted(stream, kind),
        )
