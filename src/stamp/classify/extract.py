'''
STAMP: subvolume extraction
'''

# Import external dependencies
import mrcfile, numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.ndimage import map_coordinates

# Import STAMP schema
from stamp.schemas.particles import Particle
from stamp.schemas.subvolumes import Subvolumes
from stamp.utils.errors import StampValidationError
from stamp.utils.log import log
from stamp.utils.parallel import run_parallel

# _ExtractTomogramJob: one tomogram's particle group, picklable for ProcessPoolExecutor
@dataclass(frozen=True)
class _ExtractTomogramJob:
    tomogram_id: str
    group: list[Particle]
    tomogram_path: str
    segmentation_path: str | None
    box_voxels: int
    cache_dir: Path

# quaternion_to_matrix: return a rotation matrix from a unit quaternion (w, x, y, z)
def quaternion_to_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )

# canonical_grid: sampling grid for a cubic box centred on the origin
def canonical_grid(box_voxels: int) -> np.ndarray:
    if box_voxels < 2:
        raise ValueError('box_voxels must be at least 2')
    half = (box_voxels - 1) / 2.0
    axis = np.arange(box_voxels) - half
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return np.stack([x.ravel(), y.ravel(), z.ravel()])

# extract_subvolume: sample one subvolume in the particle's canonical frame
def extract_subvolume(
    tomogram: np.ndarray,
    position_xyz: tuple[float, float, float],
    orientation: tuple[float, float, float, float] | None,
    box_voxels: int,
) -> np.ndarray:
    grid_xyz = canonical_grid(box_voxels)
    if orientation is not None:
        grid_xyz = quaternion_to_matrix(orientation) @ grid_xyz

    sample_xyz = grid_xyz + np.array(position_xyz)[:, None]
    sample_zyx = sample_xyz[::-1]  # single xyz -> zyx conversion

    values = map_coordinates(
        tomogram.astype(np.float32), sample_zyx, order=1, mode='constant', cval=0.0
    )
    return values.reshape(box_voxels, box_voxels, box_voxels)

# box_fits_inside: whether a rotated box at this position stays inside the volume
def box_fits_inside(
    position_xyz: tuple[float, float, float], shape_zyx: tuple[int, ...], box_voxels: int
) -> bool:
    radius = (box_voxels - 1) / 2.0 * np.sqrt(3.0)
    extent_xyz = np.array(shape_zyx)[::-1] - 1
    position = np.array(position_xyz)
    return bool(np.all(position >= radius) and np.all(position <= extent_xyz - radius))

# _extract_one_tomogram: worker entry point — loads one tomogram and extracts subvolumes for its particles
def _extract_one_tomogram(job: _ExtractTomogramJob) -> tuple[list[Particle], list[Particle]]:
    with mrcfile.open(str(job.tomogram_path), permissive=True) as mrc:
        tomogram = np.asarray(mrc.data).astype(np.float32)

    if job.segmentation_path is not None:
        with mrcfile.open(str(job.segmentation_path), permissive=True) as mrc:
            segmentation = np.asarray(mrc.data)
        if segmentation.shape != tomogram.shape:
            raise StampValidationError(f'segmentation shape {segmentation.shape} does not match tomogram shape {tomogram.shape} for {job.tomogram_id!r}; they must be the same volume at the same binning')
        # replace membrane voxels with the background median so the class average is not dominated by the membrane slab
        tomogram[segmentation > 0] = np.median(tomogram[segmentation <= 0])

    kept: list[Particle] = []
    skipped: list[Particle] = []
    subvolumes: list[np.ndarray] = []
    for particle in job.group:
        if particle.orientation is None or not box_fits_inside(particle.position, tomogram.shape, job.box_voxels):
            skipped.append(particle)
            continue
        subvolumes.append(extract_subvolume(tomogram, particle.position, particle.orientation, job.box_voxels))
        kept.append(particle)
    if subvolumes:
        job.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(job.cache_dir / f'{job.tomogram_id}.npy', np.stack(subvolumes))
    return kept, skipped

# extract_particle_set: extract subvolumes for a list of particles
# extract_particle_set: extract subvolumes for a list of particles, cached per tomogram
def extract_particle_set(
    particles: list[Particle],
    tomogram_paths: dict[str, str],
    box_voxels: int,
    segmentation_paths: dict[str, str] | None = None,
    n_workers: int = 1,
    *,
    cache_dir: Path,
) -> tuple[Subvolumes, list[Particle], list[Particle]]:
    by_tomogram: dict[str, list[Particle]] = {}
    for particle in particles:
        by_tomogram.setdefault(particle.tomogram_id, []).append(particle)

    segmentation_paths = segmentation_paths or {}
    kept: list[Particle] = []
    skipped: list[Particle] = []

    jobs: list[_ExtractTomogramJob] = []
    for tomogram_id, group in by_tomogram.items():
        path = tomogram_paths.get(tomogram_id)
        if path is None:
            log.warning(f'{tomogram_id}: no matching raw tomogram path, skipping {len(group)} particle(s)')
            skipped.extend(group)
            continue
        jobs.append(_ExtractTomogramJob(tomogram_id, group, path, segmentation_paths.get(tomogram_id), box_voxels, cache_dir))

    result_by_tomogram: dict[str, tuple[list[Particle], list[Particle]]] = {}

    def _on_success(job: _ExtractTomogramJob, result: tuple[list[Particle], list[Particle]]) -> None:
        result_by_tomogram[job.tomogram_id] = result

    def _on_error(job: _ExtractTomogramJob, exc: Exception) -> None:
        if isinstance(exc, StampValidationError):
            log.error(f'{job.tomogram_id}: {exc}')
            skipped.extend(job.group)
        else:
            raise exc

    run_parallel(jobs, _extract_one_tomogram, max_workers=n_workers, label='subvolume-extraction', on_success=_on_success, on_error=_on_error)

    index: list[tuple[str, int]] = []
    for job in jobs:
        result = result_by_tomogram.get(job.tomogram_id)
        if result is None:
            continue
        tomogram_kept, tomogram_skipped = result
        kept.extend(tomogram_kept)
        skipped.extend(tomogram_skipped)
        index.extend((job.tomogram_id, row) for row in range(len(tomogram_kept)))

    log.debug(f'extract_particle_set: {len(kept)} kept, {len(skipped)} skipped')
    return Subvolumes(cache_dir, index, box_voxels), kept, skipped
