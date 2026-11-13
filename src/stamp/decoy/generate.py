'''
STAMP: decoy generation
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial import cKDTree

# Import internal STAMP objects
from stamp.utils.halfset import assign_half_sets
from stamp.picking.geometry import extract_surface, quaternion_from_reference_to
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.particles import OrientationPrior, Particle, ParticleSet
from stamp.schemas.picks import RawPick
from stamp.utils.checkpoint import atomic_save_npz
from stamp.utils.errors import StampValidationError
from stamp.utils.io import cache_tomogram_picks, load_cached_picks, new_mrc, open_mrc, read_mrc
from stamp.utils.log import log
from stamp.utils.parallel import run_parallel, run_parallel_ordered

# Define constants
METHOD_SHIFTED = 'shifted'
METHOD_SYNTHETIC_NOISE = 'synthetic-noise'
METHOD_RANDOM_VOLUME = 'random-volume'
DECOY_SOURCE_PREFIX = 'decoy-'

# DecoyGeometry: voxel size and density sign shared by the decoy generators
@dataclass(frozen=True)
class DecoyGeometry:
    voxel_size_angstrom: float
    # +1 if the synthetic membrane shell is brighter than background, -1 if darker
    density_sign: int = -1

    def __post_init__(self) -> None:
        if self.voxel_size_angstrom <= 0:
            raise StampValidationError('voxel_size_angstrom must be positive.')
        if self.density_sign not in (1, -1):
            raise StampValidationError('density_sign must be +1 or -1.')

    # to_voxels: convert a distance in Angstrom to voxels
    def to_voxels(self, angstrom: float) -> float:
        return angstrom / self.voxel_size_angstrom

# _extract_surface_for_manifest: worker entry point, loads segmentation and extracts its surface
def _extract_surface_for_manifest(manifest: TomogramManifest) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    segmentation = read_mrc(manifest.segmentation_path)
    vertices, normals = extract_surface(segmentation)
    return segmentation.shape, vertices, normals

# _save_cached_surface: checkpoint one tomogram's extracted surface
def _save_cached_surface(surfaces_dir: Path, tomogram_id: str, surface: tuple[tuple[int, ...], np.ndarray, np.ndarray]) -> None:
    shape, vertices, normals = surface
    atomic_save_npz(surfaces_dir / f'{tomogram_id}.npz', shape=np.array(shape), vertices=vertices, normals=normals)

# _load_cached_surface: one tomogram's checkpointed surface, or None when absent or unreadable
def _load_cached_surface(surfaces_dir: Path | None, tomogram_id: str) -> tuple[tuple[int, ...], np.ndarray, np.ndarray] | None:
    if surfaces_dir is None:
        return None
    try:
        with np.load(surfaces_dir / f'{tomogram_id}.npz') as cached:
            return tuple(int(size) for size in cached['shape']), cached['vertices'], cached['normals']
    except (OSError, ValueError, KeyError):
        return None

# generate_shifted_decoys: displace each real pick by a large random vector away from surface and picks
def generate_shifted_decoys(
    real_particle_set: ParticleSet,
    manifests: list[TomogramManifest],
    config: DecoyGeometry,
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
    surfaces_dir = output_dir / 'raw' / METHOD_SHIFTED / 'surfaces' if output_dir is not None else None
    shape_by_tomogram = {}
    surface_by_tomogram: dict[str, tuple[cKDTree, np.ndarray]] = {}

    # _register: index one tomogram's surface for the sampling loop below
    def _register(manifest: TomogramManifest, result: tuple[tuple[int, ...], np.ndarray, np.ndarray]) -> None:
        shape, vertices, normals = result
        shape_by_tomogram[manifest.tomogram_id] = shape
        surface_by_tomogram[manifest.tomogram_id] = (cKDTree(vertices[:, ::-1]), normals[:, ::-1])

    def _on_success(manifest: TomogramManifest, result: tuple[tuple[int, ...], np.ndarray, np.ndarray] | None) -> None:
        if result is None:
            return
        _register(manifest, result)
        if surfaces_dir is not None:
            _save_cached_surface(surfaces_dir, manifest.tomogram_id, result)
    def _on_error(manifest: TomogramManifest, exc: Exception) -> None:
        if isinstance(exc, ValueError):
            log.warning(f'{manifest.tomogram_id}: skipped ({exc})')
        else:
            raise exc

    pending: list[TomogramManifest] = []
    for manifest in manifests:
        cached = _load_cached_surface(surfaces_dir, manifest.tomogram_id)
        if cached is None:
            pending.append(manifest)
        else:
            _register(manifest, cached)
    if len(pending) < len(manifests):
        log.info(f'Resuming shifted decoys: {len(manifests) - len(pending)} tomogram(s) loaded from checkpoint')
    run_parallel(pending, _extract_surface_for_manifest, max_workers=n_workers, label='decoy-surface-extract', on_success=_on_success, on_error=_on_error)

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

# generate_random_volume_decoys: one decoy per real pick at a uniformly random position in the tomogram, clear of every real pick and of the border, with a random orientation where the prior has one; needs no segmentation so it serves every prior except the membrane normal
def generate_random_volume_decoys(
    real_particle_set: ParticleSet,
    manifests: list[TomogramManifest],
    config: DecoyGeometry,
    min_distance_from_picks_angstrom: float,
    border_angstrom: float,
    seed: int,
    max_attempts_per_particle: int = 200,
) -> ParticleSet | None:
    prior = real_particle_set.orientation_prior
    if prior is OrientationPrior.NORMAL:
        raise StampValidationError('random-volume decoys have no membrane normal to orient by, use the shifted method for the normal orientation prior')
    log.progress('Generating random-volume decoys')
    rng = np.random.default_rng(seed)
    raw_path_by_tomogram = {manifest.tomogram_id: manifest.raw_tomogram_path for manifest in manifests}
    positions_by_tomogram: dict[str, list[tuple[float, float, float]]] = {}
    for particle in real_particle_set.particles:
        positions_by_tomogram.setdefault(particle.tomogram_id, []).append(particle.position)
    min_distance = config.to_voxels(min_distance_from_picks_angstrom)
    border = config.to_voxels(border_angstrom)
    raw_picks: list[RawPick] = []
    for tomogram_id, positions in positions_by_tomogram.items():
        if tomogram_id not in raw_path_by_tomogram:
            continue
        with open_mrc(raw_path_by_tomogram[tomogram_id], header_only=True) as mrc:
            extent_xyz = np.array([int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)]) - 1
        low, high = np.full(3, border), extent_xyz - border
        if np.any(high <= low):
            log.warning(f'{tomogram_id}: smaller than twice the {border_angstrom} Å border, no random-volume decoys placed')
            continue
        pick_tree = cKDTree(np.array(positions))
        for _ in positions:
            for _attempt in range(max_attempts_per_particle):
                candidate = rng.uniform(low, high)
                if pick_tree.query(candidate, k=1)[0] < min_distance:
                    continue
                axis = rng.normal(size=3)
                raw_picks.append(RawPick(
                    tomogram_id=tomogram_id,
                    position=tuple(float(c) for c in candidate),
                    orientation=quaternion_from_reference_to(axis / np.linalg.norm(axis)) if prior is OrientationPrior.AXIS else None,
                    source_picker=f'{DECOY_SOURCE_PREFIX}{METHOD_RANDOM_VOLUME}',
                ))
                break
            else:
                log.debug(f'{tomogram_id}: exhausted {max_attempts_per_particle} attempts placing a random-volume decoy, skipping one particle')
    log.info(f'Generated {len(raw_picks)} decoy particles')
    return _finalise(raw_picks, seed, METHOD_RANDOM_VOLUME, prior)

# _SyntheticNoiseJob: one noise volume's inputs
@dataclass(frozen=True)
class _SyntheticNoiseJob:
    index: int
    tomogram_shape: tuple[int, int, int]
    shell: np.ndarray
    centre: np.ndarray
    segmentation_dir: Path
    tomogram_dir: Path
    config: DecoyGeometry
    seed: int
    n_decoys_per_tomogram: int
    cache_dir: Path | None = None

# _generate_one_synthetic_noise_tomogram: multiprocessing worker entry point, reusing a finished volume from the checkpoint
def _generate_one_synthetic_noise_tomogram(job: _SyntheticNoiseJob) -> tuple[TomogramManifest, list[RawPick]]:
    tomogram_id = f'decoy-noise-{job.index:03d}'
    segmentation_path = job.segmentation_dir / f'{tomogram_id}.mrc'
    tomogram_path = job.tomogram_dir / f'{tomogram_id}.mrc'
    manifest = TomogramManifest(
        tomogram_id=tomogram_id,
        segmentation_path=segmentation_path,
        raw_tomogram_path=tomogram_path,
        voxel_size_angstrom=job.config.voxel_size_angstrom,
        is_decoy=True,
    )

    # the pick cache is written last and atomically, so its presence means the volumes and picks are complete
    if job.cache_dir is not None and (job.cache_dir / f'{tomogram_id}.json').is_file() and segmentation_path.is_file() and tomogram_path.is_file():
        log.debug(f'{tomogram_id}: loaded from checkpoint')
        return manifest, load_cached_picks(job.cache_dir, [tomogram_id])

    rng = np.random.default_rng(job.seed + job.index)
    volume = rng.normal(0.0, 1.0, size=job.tomogram_shape).astype(np.float32)
    volume[job.shell > 0] += job.config.density_sign * 2.0

    new_mrc(segmentation_path, job.shell)
    new_mrc(tomogram_path, volume)

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
    config: DecoyGeometry,
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

# _finalise: assign decoy particle IDs and half-sets, returning a decoy ParticleSet or None if no decoys were generated
def _finalise(raw_picks: list[RawPick], seed: int, method: str, orientation_prior: OrientationPrior = OrientationPrior.NORMAL) -> ParticleSet | None:
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
        orientation_prior=orientation_prior,
    )
