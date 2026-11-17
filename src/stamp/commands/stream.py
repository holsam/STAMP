'''
STAMP: stream logic
'''

# Import external dependencies
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.extract import extract_particle_set
from stamp.commands.pick import pick_tomogram
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
from stamp.schemas.particles import HalfSet, OrientationPrior, Particle, ParticleSet, particle_ids
from stamp.schemas.picks import RawPick
from stamp.stream.state import StreamDir, StreamSettings, load_or_init_settings
from stamp.utils.checkpoint import atomic_write_json
from stamp.utils.errors import StampPipelineError, StampValidationError
from stamp.utils.log import log
from stamp.utils.mrc import open_mrc
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
