'''
STAMP: decoy generation
'''

# Import external dependencies
import mrcfile, numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial import cKDTree

# Import internal STAMP objects
from stamp.utils.halfset import assign_half_sets
from stamp.picking.geometry import (
    downsample_points,
    exclude_near_boundary,
    extract_surface,
    quaternion_from_reference_to,
    robust_normalise,
    sample_along_normals,
    score_membrane_faces
)
from stamp.picking.native import NativePickerConfig
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.particles import Particle, ParticleSet
from stamp.schemas.picks import RawPick
from stamp.utils.errors import StampValidationError
from stamp.utils.log import log
from stamp.utils.parallel import run_parallel, run_parallel_ordered

# Define constants
METHOD_REJECTED_SURFACE = 'rejected-surface'
METHOD_SHIFTED = 'shifted'
METHOD_SYNTHETIC_NOISE = 'synthetic-noise'
DECOY_SOURCE_PREFIX = 'decoy-'

# _RejectedSurfaceJob: one manifest's scoring inputs
@dataclass(frozen=True)
class _RejectedSurfaceJob:
    manifest: TomogramManifest
    config: NativePickerConfig

# _ScoreSurfaceResult: type alias for _score_surface's return shape
_ScoreSurfaceResult = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

# _adaptive_decoy_count: target decoy count derived from a tomogram's real pick count
def _adaptive_decoy_count(n_real_picks: int, *, target_ratio: float = 1.0) -> int:
    return max(1, round(n_real_picks * target_ratio))

# _score_one_tomogram: worker entry point, isolates StampValidationError per tomogram for on_error
def _score_one_tomogram(job: _RejectedSurfaceJob) -> _ScoreSurfaceResult:
    return _score_surface(job.manifest, job.config)

# generate_rejected_surface_decoys: sample decoys from surface points the native picker scored and rejected
def generate_rejected_surface_decoys(
    real_particle_set: ParticleSet,
    manifests: list[TomogramManifest],
    config: NativePickerConfig,
    n_decoys_per_tomogram: int | None,
    min_distance_from_real_angstrom: float,
    seed: int,
    *,
    n_workers: int = 1,
    output_dir: Path | None = None,
) -> ParticleSet | None:
    log.progress(f'Generating rejected-surface decoys ({n_decoys_per_tomogram or "adaptive n"}/tomogram)')
    rng = np.random.default_rng(seed)
    real_by_tomogram: dict[str, list[tuple[float, float, float]]] = {}
    for particle in real_particle_set.particles:
        real_by_tomogram.setdefault(particle.tomogram_id, []).append(particle.position)

    min_distance_voxels = config.to_voxels(min_distance_from_real_angstrom)
    raw_picks: list[RawPick] = []
    cache_dir = output_dir / 'raw' / METHOD_REJECTED_SURFACE if output_dir is not None else None
    tomogram_ids = []

    # run surface scoring in a pool but keep rng-driven sampling sequential and in manifest order so results stay reproducible for a given seed
    result_by_tomogram: dict[str, _ScoreSurfaceResult] = {}

    def _on_success(job: _RejectedSurfaceJob, result: _ScoreSurfaceResult) -> None:
        result_by_tomogram[job.manifest.tomogram_id] = result

    def _on_error(job: _RejectedSurfaceJob, exc: Exception) -> None:
        if isinstance(exc, StampValidationError):
            log.warning(f'{job.manifest.tomogram_id}: skipped ({exc})')
        else:
            raise exc

    jobs = [_RejectedSurfaceJob(manifest, config) for manifest in manifests]
    run_parallel(jobs, _score_one_tomogram, max_workers=n_workers, label='decoy-surface-score', on_success=_on_success, on_error=_on_error)

    for manifest in manifests:
        result = result_by_tomogram.get(manifest.tomogram_id)
        if result is None:
            continue
        points_zyx, normals_zyx, scores, winning_window, mean_scores, profile_scores = result
        if points_zyx.shape[0] == 0:
            continue

        # only use points the picker rejected
        rejected = scores < config.effective_n_mad
        points_zyx, normals_zyx, scores, winning_window, mean_scores, profile_scores = (
            points_zyx[rejected], normals_zyx[rejected], scores[rejected], winning_window[rejected],
            mean_scores[rejected], profile_scores[rejected],
        )
        if points_zyx.shape[0] == 0:
            continue

        # convert positions (z, y, x) to position (x, y, z)
        points_xyz = points_zyx[:, ::-1]
        real_positions = real_by_tomogram.get(manifest.tomogram_id, [])
        if real_positions:
            tree = cKDTree(np.array(real_positions))
            far_enough = (
                tree.query(points_xyz, k=1)[0] >= min_distance_voxels
            )
            points_xyz, points_zyx, normals_zyx, scores, winning_window, mean_scores, profile_scores = (
                points_xyz[far_enough], points_zyx[far_enough],
                normals_zyx[far_enough], scores[far_enough], winning_window[far_enough],
                mean_scores[far_enough], profile_scores[far_enough],
            )
        if points_xyz.shape[0] == 0:
            continue

        # prefer the most confidently empty positions, but jitter the choice so decoys aren't all clustered in one flat region
        target = n_decoys_per_tomogram if n_decoys_per_tomogram is not None else _adaptive_decoy_count(len(real_positions))
        n_to_take = min(target, points_xyz.shape[0])
        candidate_pool = min(points_xyz.shape[0], n_to_take * 5)
        lowest_first = np.argsort(scores)[:candidate_pool]
        chosen = rng.choice(lowest_first, size=n_to_take, replace=False)

        tomogram_picks = [
            RawPick(
                tomogram_id=manifest.tomogram_id,
                position=tuple(float(c) for c in points_xyz[index]),
                orientation=quaternion_from_reference_to(np.asarray(normals_zyx[index][::-1])),
                confidence=float(scores[index]),
                source_picker=f'{DECOY_SOURCE_PREFIX}{METHOD_REJECTED_SURFACE}',
                offset_window_angstrom=config.offset_windows_angstrom[int(winning_window[index])],
                mean_score=float(mean_scores[index]),
                profile_score=float(profile_scores[index]),
            )
            for index in chosen
        ]
        if cache_dir is not None:
            cache_tomogram_picks(cache_dir, manifest.tomogram_id, tomogram_picks)
            tomogram_ids.append(manifest.tomogram_id)
        else:
            raw_picks.extend(tomogram_picks)

    if cache_dir is not None:
        raw_picks = load_cached_picks(cache_dir, tomogram_ids)
    decoys = _finalise(raw_picks, seed, METHOD_REJECTED_SURFACE)
    log.info(f'Generated {len(raw_picks)} decoy particles')
    return decoys

# _extract_surface_for_manifest: worker entry point, loads segmentation and extracts its surface
def _extract_surface_for_manifest(manifest: TomogramManifest) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    with mrcfile.open(str(manifest.segmentation_path), permissive=True) as mrc:
        segmentation = np.asarray(mrc.data)
    vertices, normals = extract_surface(segmentation)
    return segmentation.shape, vertices, normals

# generate_shifted_decoys: displace each real pick by a large random vector away from surface and picks
def generate_shifted_decoys(
    real_particle_set: ParticleSet,
    manifests: list[TomogramManifest],
    config: NativePickerConfig,
    min_shift_angstrom: float,
    max_shift_angstrom: float,
    min_distance_from_surface_angstrom: float,
    min_pick_distance: float,
    seed: int,
    max_attempts_per_particle: int = 200,
    *,
    n_workers: int = 1,
    output_dir: Path | None = None,
) -> ParticleSet | None:
    log.progress('Generating shifted decoys')
    rng = np.random.default_rng(seed)
    shape_by_tomogram = {}
    surface_by_tomogram: dict[str, tuple[cKDTree, np.ndarray]] = {}
    def _on_success(manifest: TomogramManifest, result: tuple[tuple[int, ...], np.ndarray, np.ndarray] | None) -> None:
        if result is None:
            return
        shape, vertices, normals = result
        shape_by_tomogram[manifest.tomogram_id] = shape
        surface_by_tomogram[manifest.tomogram_id] = (cKDTree(vertices[:, ::-1]), normals[:, ::-1])
    def _on_error(manifest: TomogramManifest, exc: Exception) -> None:
        if isinstance(exc, ValueError):
            log.warning(f'{manifest.tomogram_id}: skipped ({exc})')
        else:
            raise exc

    run_parallel(manifests, _extract_surface_for_manifest, max_workers=n_workers, label='decoy-surface-extract', on_success=_on_success, on_error=_on_error)

    real_by_tomogram: dict[str, list[tuple[float, float, float]]] = {}
    for particle in real_particle_set.particles:
        real_by_tomogram.setdefault(particle.tomogram_id, []).append(particle.position)

    min_shift = config.to_voxels(min_shift_angstrom)
    max_shift = config.to_voxels(max_shift_angstrom)
    min_surface_distance = config.to_voxels(min_distance_from_surface_angstrom)
    min_pick_distance_voxels = config.to_voxels(min_pick_distance)

    raw_picks: list[RawPick] = []
    cache_dir = output_dir / 'raw' / METHOD_SHIFTED if output_dir is not None else None
    tomogram_ids: list[str] = []
    for tomogram_id, positions in real_by_tomogram.items():
        shape = shape_by_tomogram.get(tomogram_id)
        surface = surface_by_tomogram.get(tomogram_id)
        if shape is None or surface is None:
            continue
        surface_tree, surface_normals_xyz = surface
        pick_tree = cKDTree(np.array(positions))
        extent_xyz = np.array(shape)[::-1] - 1

        tomogram_picks: list[RawPick] = []
        for position in positions:
            origin = np.array(position)
            for _attempt in range(max_attempts_per_particle):
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                magnitude = rng.uniform(min_shift, max_shift)
                candidate = origin + direction * magnitude
                surface_distance, nearest_vertex = surface_tree.query(candidate, k=1)

                if np.any(candidate < 0) or np.any(candidate > extent_xyz):
                    continue
                if surface_distance < min_surface_distance:
                    continue
                if pick_tree.query(candidate, k=1)[0] < min_pick_distance_voxels:
                    continue

                tomogram_picks.append(
                    RawPick(
                        tomogram_id=tomogram_id,
                        position=tuple(float(c) for c in candidate),
                        orientation=quaternion_from_reference_to(surface_normals_xyz[nearest_vertex]),
                        confidence=None,
                        source_picker=f'{DECOY_SOURCE_PREFIX}{METHOD_SHIFTED}',
                    )
                )
                break
            else:
                log.debug(f'{tomogram_id}: exhausted {max_attempts_per_particle} attempts placing a shifted decoy, skipping one particle')

        if cache_dir is not None:
            cache_tomogram_picks(cache_dir, tomogram_id, tomogram_picks)
            tomogram_ids.append(tomogram_id)
        else:
            raw_picks.extend(tomogram_picks)

    if cache_dir is not None:
        raw_picks = load_cached_picks(cache_dir, tomogram_ids)
    decoys = _finalise(raw_picks, seed, METHOD_SHIFTED)
    log.info(f'Generated {len(raw_picks)} decoy particles')
    return decoys

# _SyntheticNoiseJob: one noise volume's inputs
@dataclass(frozen=True)
class _SyntheticNoiseJob:
    index: int
    tomogram_shape: tuple[int, int, int]
    shell: np.ndarray
    centre: np.ndarray
    segmentation_dir: Path
    tomogram_dir: Path
    config: NativePickerConfig
    seed: int
    n_decoys_per_tomogram: int
    cache_dir: Path | None = None

# _generate_one_synthetic_noise_tomogram: multiprocessing worker entry point
def _generate_one_synthetic_noise_tomogram(job: _SyntheticNoiseJob) -> tuple[TomogramManifest, list[RawPick]]:
    rng = np.random.default_rng(job.seed + job.index)
    tomogram_id = f'decoy-noise-{job.index:03d}'
    volume = rng.normal(0.0, 1.0, size=job.tomogram_shape).astype(np.float32)
    volume[job.shell > 0] += job.config.density_sign * 2.0

    segmentation_path = job.segmentation_dir / f'{tomogram_id}.mrc'
    tomogram_path = job.tomogram_dir / f'{tomogram_id}.mrc'
    with mrcfile.new(segmentation_path, overwrite=True) as mrc:
        mrc.set_data(job.shell)
    with mrcfile.new(tomogram_path, overwrite=True) as mrc:
        mrc.set_data(volume)

    manifest = TomogramManifest(
        tomogram_id=tomogram_id,
        segmentation_path=segmentation_path,
        raw_tomogram_path=tomogram_path,
        voxel_size_angstrom=job.config.voxel_size_angstrom,
        is_decoy=True,
    )

    shell_voxels = np.argwhere(job.shell > 0)
    chosen = rng.choice(
        shell_voxels.shape[0],
        size=min(job.n_decoys_per_tomogram, shell_voxels.shape[0]),
        replace=False,
    )
    raw_picks = []
    for voxel_index in chosen:
        position_zyx = shell_voxels[voxel_index].astype(float)
        radial_xyz = (position_zyx - job.centre)[::-1]
        raw_picks.append(
            RawPick(
                tomogram_id=tomogram_id,
                position=tuple(float(c) for c in position_zyx[::-1]),
                orientation=quaternion_from_reference_to(radial_xyz),
                confidence=None,
                source_picker=f'{DECOY_SOURCE_PREFIX}{METHOD_SYNTHETIC_NOISE}',
            )
        )
    if job.cache_dir is not None:
        cache_tomogram_picks(job.cache_dir, tomogram_id, raw_picks)
    return manifest, raw_picks

# generate_synthetic_noise_decoys: Gaussian-noise volumes with a membrane-like shell and shell-sampled decoys
def generate_synthetic_noise_decoys(
    tomogram_shape: tuple[int, int, int],
    n_tomograms: int,
    n_decoys_per_tomogram: int | None,
    output_dir: Path,
    config: NativePickerConfig,
    seed: int,
    *,
    n_workers: int = 1,
) -> tuple[ParticleSet | None, list[TomogramManifest]]:
    if n_decoys_per_tomogram is None:
        raise StampValidationError('n_decoys_per_tomogram must be set explicitly for synthetic-noise decoys')
    log.progress(f'Generating synthetic-noise decoys ({n_tomograms} tomograms, {n_decoys_per_tomogram}/tomogram)')
    segmentation_dir = output_dir / 'segmentations'
    tomogram_dir = output_dir / 'tomograms'
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    tomogram_dir.mkdir(parents=True, exist_ok=True)

    centre = np.array(tomogram_shape) / 2.0
    radius = float(min(tomogram_shape)) / 3.0
    grid = np.stack(
        np.meshgrid(*[np.arange(s) for s in tomogram_shape], indexing='ij'), axis=-1
    )
    distance = np.linalg.norm(grid - centre, axis=-1)
    shell = ((distance > radius - 1.5) & (distance < radius + 1.5)).astype(np.float32)
    cache_dir = output_dir / 'raw' / METHOD_SYNTHETIC_NOISE
    jobs = [_SyntheticNoiseJob(index, tomogram_shape, shell, centre, segmentation_dir, tomogram_dir, config, seed, n_decoys_per_tomogram, cache_dir) for index in range(n_tomograms)]
    per_tomogram_results = run_parallel_ordered(jobs, _generate_one_synthetic_noise_tomogram, max_workers=n_workers, label='decoy-synthetic-noise')

    manifests = [manifest for manifest, _ in per_tomogram_results]
    raw_picks = load_cached_picks(cache_dir, [manifest.tomogram_id for manifest in manifests])

    decoys = _finalise(raw_picks, seed, METHOD_SYNTHETIC_NOISE)
    log.info(f'Generated {len(raw_picks)} decoy particles')
    return decoys, manifests

# _score_surface: re-run the native picker's surface scoring for one tomogram
def _score_surface(
    manifest: TomogramManifest, config: NativePickerConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with mrcfile.open(str(manifest.segmentation_path), permissive=True) as mrc:
        segmentation = np.asarray(mrc.data)
    with mrcfile.open(str(manifest.raw_tomogram_path), permissive=True) as mrc:
        tomogram = np.asarray(mrc.data)

    if segmentation.shape != tomogram.shape:
        raise StampValidationError(f'Segmentation and tomogram shapes disagree for {manifest.tomogram_id}')

    try:
        vertices, normals = extract_surface(segmentation)
    except ValueError as exc:
        log.warning(f'{manifest.tomogram_id}: skipped ({exc})')
        empty = np.empty((0, 3))
        return empty, empty, np.empty(0), np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)

    vertices, normals = downsample_points(
        vertices, normals, config.to_voxels(config.surface_spacing_angstrom)
    )
    margin = (
        config.boundary_margin_angstrom
        if config.boundary_margin_angstrom is not None
        else config.max_offset_angstrom
    )
    inside = exclude_near_boundary(vertices, segmentation.shape, config.to_voxels(margin))
    vertices, normals = vertices[inside], normals[inside]
    if vertices.shape[0] == 0:
        empty = np.empty((0, 3))
        return empty, empty, np.empty(0), np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)

    offset_windows_voxels = [(config.to_voxels(offset_min), config.to_voxels(offset_max)) for offset_min, offset_max in config.offset_windows_angstrom]
    profile_width_voxels = (config.to_voxels(config.profile_width_angstrom) if config.profile_width_angstrom is not None else None)

    points, normals_out, scores, winning_window, _used_fallback, mean_scores, profile_scores = score_membrane_faces(
        tomogram, vertices, normals, offset_windows_voxels, config.n_samples, config.density_sign,
        scoring_mode=config.scoring_mode, profile_width_voxels=profile_width_voxels,
    )
    return points, normals_out, scores, winning_window, mean_scores, profile_scores

# _finalise: assign decoy particle IDs and half-sets, returning a decoy ParticleSet or None if no decoys were generated
def _finalise(raw_picks: list[RawPick], seed: int, method: str) -> ParticleSet | None:
    particle_ids = [f'decoy-{index:06d}' for index in range(len(raw_picks))]
    if not particle_ids:
        return None
    group_of = {particle_id: pick.tomogram_id for particle_id, pick in zip(particle_ids, raw_picks)}
    half_set_by_id = assign_half_sets(particle_ids, seed=seed, group_of=group_of)
    particles = [
        Particle(
            particle_id=particle_id,
            tomogram_id=pick.tomogram_id,
            position=pick.position,
            orientation=pick.orientation,
            source_picker=pick.source_picker,
            confidence=pick.confidence,
            half_set=half_set_by_id[particle_id],
        )
        for particle_id, pick in zip(particle_ids, raw_picks)
    ]
    return ParticleSet(
        particles=particles,
        consensus_rule='union',
        contributing_pickers=[f'{DECOY_SOURCE_PREFIX}{method}'],
    )
