'''
STAMP: decoy generation
'''

# Import external dependencies
import mrcfile, numpy as np
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
)
from stamp.picking.native import NativePickerConfig
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.particles import Particle, ParticleSet
from stamp.schemas.picks import RawPick

# Define constants
METHOD_REJECTED_SURFACE = 'rejected-surface'
METHOD_SHIFTED = 'shifted'
METHOD_SYNTHETIC_NOISE = 'synthetic-noise'
DECOY_SOURCE_PREFIX = 'decoy-'

# generate_rejected_surface_decoys: sample decoys from surface points the native picker scored and rejected
def generate_rejected_surface_decoys(
    real_particle_set: ParticleSet,
    manifests: list[TomogramManifest],
    config: NativePickerConfig,
    n_decoys_per_tomogram: int,
    min_distance_from_real_angstrom: float,
    seed: int,
) -> ParticleSet | None:
    rng = np.random.default_rng(seed)
    real_by_tomogram: dict[str, list[tuple[float, float, float]]] = {}
    for particle in real_particle_set.particles:
        real_by_tomogram.setdefault(particle.tomogram_id, []).append(particle.position)

    min_distance_voxels = config.to_voxels(min_distance_from_real_angstrom)
    raw_picks: list[RawPick] = []

    for manifest in manifests:
        points_zyx, normals_zyx, scores = _score_surface(manifest, config)
        if points_zyx.shape[0] == 0:
            continue

        # only use points the picker rejected
        rejected = scores < config.n_mad
        points_zyx, normals_zyx, scores = (
            points_zyx[rejected], normals_zyx[rejected], scores[rejected]
        )
        if points_zyx.shape[0] == 0:
            continue

        # convert positions (x, y, z) to geometry (z, y, x)
        points_xyz = points_zyx[:, ::-1]
        real_positions = real_by_tomogram.get(manifest.tomogram_id, [])
        if real_positions:
            tree = cKDTree(np.array(real_positions))
            far_enough = (
                tree.query(points_xyz, k=1)[0] >= min_distance_voxels
            )
            points_xyz, points_zyx, normals_zyx, scores = (
                points_xyz[far_enough], points_zyx[far_enough],
                normals_zyx[far_enough], scores[far_enough],
            )
        if points_xyz.shape[0] == 0:
            continue

        # prefer the most confidently empty positions, but jitter the choice so decoys aren't all clustered in one flat region
        n_to_take = min(n_decoys_per_tomogram, points_xyz.shape[0])
        candidate_pool = min(points_xyz.shape[0], n_to_take * 5)
        lowest_first = np.argsort(scores)[:candidate_pool]
        chosen = rng.choice(lowest_first, size=n_to_take, replace=False)

        for index in chosen:
            normal_xyz = normals_zyx[index][::-1]
            raw_picks.append(
                RawPick(
                    tomogram_id=manifest.tomogram_id,
                    position=tuple(float(c) for c in points_xyz[index]),
                    orientation=quaternion_from_reference_to(np.asarray(normal_xyz)),
                    confidence=float(scores[index]),
                    source_picker=f'{DECOY_SOURCE_PREFIX}{METHOD_REJECTED_SURFACE}',
                )
            )

    return _finalise(raw_picks, seed, METHOD_REJECTED_SURFACE)

# generate_shifted_decoys: displace each real pick by a large random vector away from surface and picks
def generate_shifted_decoys(
    real_particle_set: ParticleSet,
    manifests: list[TomogramManifest],
    config: NativePickerConfig,
    min_shift_angstrom: float,
    max_shift_angstrom: float,
    min_distance_from_surface_angstrom: float,
    seed: int,
    max_attempts_per_particle: int = 200,
) -> ParticleSet | None:
    rng = np.random.default_rng(seed)
    shape_by_tomogram = {}

    surface_by_tomogram: dict[str, tuple[cKDTree, np.ndarray]] = {}
    for manifest in manifests:
        with mrcfile.open(str(manifest.segmentation_path), permissive=True) as mrc:
            segmentation = np.asarray(mrc.data)
        shape_by_tomogram[manifest.tomogram_id] = segmentation.shape
        try:
            vertices, normals = extract_surface(segmentation)
        except ValueError:
            continue
        surface_by_tomogram[manifest.tomogram_id] = (cKDTree(vertices[:, ::-1]), normals[:, ::-1])

    real_by_tomogram: dict[str, list[tuple[float, float, float]]] = {}
    for particle in real_particle_set.particles:
        real_by_tomogram.setdefault(particle.tomogram_id, []).append(particle.position)

    min_shift = config.to_voxels(min_shift_angstrom)
    max_shift = config.to_voxels(max_shift_angstrom)
    min_surface_distance = config.to_voxels(min_distance_from_surface_angstrom)

    raw_picks: list[RawPick] = []
    for tomogram_id, positions in real_by_tomogram.items():
        shape = shape_by_tomogram.get(tomogram_id)
        surface = surface_by_tomogram.get(tomogram_id)
        if shape is None or surface is None:
            continue
        surface_tree, surface_normals_xyz = surface
        pick_tree = cKDTree(np.array(positions))
        extent_xyz = np.array(shape)[::-1] - 1

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
                if pick_tree.query(candidate, k=1)[0] < min_surface_distance:
                    continue

                raw_picks.append(
                    RawPick(
                        tomogram_id=tomogram_id,
                        position=tuple(float(c) for c in candidate),
                        orientation=quaternion_from_reference_to(surface_normals_xyz[nearest_vertex]),
                        confidence=None,
                        source_picker=f'{DECOY_SOURCE_PREFIX}{METHOD_SHIFTED}',
                    )
                )
                break

    return _finalise(raw_picks, seed, METHOD_SHIFTED)

# generate_synthetic_noise_decoys: Gaussian-noise volumes with a membrane-like shell and shell-sampled decoys
def generate_synthetic_noise_decoys(
    tomogram_shape: tuple[int, int, int],
    n_tomograms: int,
    n_decoys_per_tomogram: int,
    output_dir: Path,
    config: NativePickerConfig,
    seed: int,
) -> tuple[ParticleSet | None, list[TomogramManifest]]:
    rng = np.random.default_rng(seed)
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

    manifests: list[TomogramManifest] = []
    raw_picks: list[RawPick] = []

    for index in range(n_tomograms):
        tomogram_id = f'decoy-noise-{index:03d}'
        volume = rng.normal(0.0, 1.0, size=tomogram_shape).astype(np.float32)
        volume[shell > 0] += config.density_sign * 2.0

        segmentation_path = segmentation_dir / f'{tomogram_id}.mrc'
        tomogram_path = tomogram_dir / f'{tomogram_id}.mrc'
        with mrcfile.new(segmentation_path, overwrite=True) as mrc:
            mrc.set_data(shell)
        with mrcfile.new(tomogram_path, overwrite=True) as mrc:
            mrc.set_data(volume)

        manifests.append(
            TomogramManifest(
                tomogram_id=tomogram_id,
                segmentation_path=segmentation_path,
                raw_tomogram_path=tomogram_path,
                voxel_size_angstrom=config.voxel_size_angstrom,
                is_decoy=True,
            )
        )

        shell_voxels = np.argwhere(shell > 0)
        chosen = rng.choice(
            shell_voxels.shape[0],
            size=min(n_decoys_per_tomogram, shell_voxels.shape[0]),
            replace=False,
        )
        for voxel_index in chosen:
            position_zyx = shell_voxels[voxel_index].astype(float)
            radial_xyz = (position_zyx - centre)[::-1]
            raw_picks.append(
                RawPick(
                    tomogram_id=tomogram_id,
                    position=tuple(float(c) for c in position_zyx[::-1]),
                    orientation=quaternion_from_reference_to(radial_xyz),
                    confidence=None,
                    source_picker=f'{DECOY_SOURCE_PREFIX}{METHOD_SYNTHETIC_NOISE}',
                )
            )

    return _finalise(raw_picks, seed, METHOD_SYNTHETIC_NOISE), manifests

# _score_surface: re-run the native picker's surface scoring for one tomogram
def _score_surface(
    manifest: TomogramManifest, config: NativePickerConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with mrcfile.open(str(manifest.segmentation_path), permissive=True) as mrc:
        segmentation = np.asarray(mrc.data)
    with mrcfile.open(str(manifest.raw_tomogram_path), permissive=True) as mrc:
        tomogram = np.asarray(mrc.data)

    if segmentation.shape != tomogram.shape:
        raise ValueError(f'Segmentation and tomogram shapes disagree for {manifest.tomogram_id}')

    try:
        vertices, normals = extract_surface(segmentation)
    except ValueError:
        empty = np.empty((0, 3))
        return empty, empty, np.empty(0)

    vertices, normals = downsample_points(
        vertices, normals, config.to_voxels(config.surface_spacing_angstrom)
    )
    margin = (
        config.boundary_margin_angstrom
        if config.boundary_margin_angstrom is not None
        else config.offset_max_angstrom
    )
    inside = exclude_near_boundary(vertices, segmentation.shape, config.to_voxels(margin))
    vertices, normals = vertices[inside], normals[inside]
    if vertices.shape[0] == 0:
        empty = np.empty((0, 3))
        return empty, empty, np.empty(0)

    offset_min = config.to_voxels(config.offset_min_angstrom)
    offset_max = config.to_voxels(config.offset_max_angstrom)

    points_parts, normals_parts, scores_parts = [], [], []
    for direction in (1, -1):
        densities = sample_along_normals(
            tomogram, vertices, normals, offset_min, offset_max,
            config.n_samples, direction,
        )
        points_parts.append(vertices)
        normals_parts.append(direction * normals)
        scores_parts.append(robust_normalise(config.density_sign * densities))

    return (
        np.concatenate(points_parts),
        np.concatenate(normals_parts),
        np.concatenate(scores_parts),
    )


# _finalise: assign decoy particle IDs and half-sets, returning a decoy ParticleSet or None if no decoys were generated
def _finalise(raw_picks: list[RawPick], seed: int, method: str) -> ParticleSet | None:
    particle_ids = [f'decoy-{index:06d}' for index in range(len(raw_picks))]
    if not particle_ids:
        return None

    half_set_by_id = assign_half_sets(particle_ids, seed=seed)
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
