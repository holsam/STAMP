'''
STAMP: subvolume extraction
'''

# Import external dependencies
import json, mrcfile, numpy as np, os, shutil
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
    box_voxels: int
    cache_dir: Path
    subtracted: bool = False

# _SubtractTomogramJob: one tomogram's membrane subtraction, written to disk for extraction to mmap
@dataclass(frozen=True)
class _SubtractTomogramJob:
    tomogram_id: str
    tomogram_path: str
    segmentation_path: str
    output_path: Path

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

    values = map_coordinates(np.asarray(tomogram, dtype=np.float32), sample_zyx, order=1, mode='constant', cval=0.0)
    return values.reshape(box_voxels, box_voxels, box_voxels)

# box_fits_inside: whether a rotated box at this position stays inside the volume
def box_fits_inside(
    position_xyz: tuple[float, float, float], shape_zyx: tuple[int, ...], box_voxels: int
) -> bool:
    radius = (box_voxels - 1) / 2.0 * np.sqrt(3.0)
    extent_xyz = np.array(shape_zyx)[::-1] - 1
    position = np.array(position_xyz)
    return bool(np.all(position >= radius) and np.all(position <= extent_xyz - radius))

# _subtract_one_tomogram: worker entry point to replace membrane voxels with the background median and write a float32 mrc
def _subtract_one_tomogram(job: _SubtractTomogramJob) -> Path:
    with mrcfile.open(job.tomogram_path, permissive=True) as mrc:
        tomogram = np.array(mrc.data, dtype=np.float32)  # writable copy
        voxel_size = mrc.voxel_size.copy()
    with mrcfile.open(job.segmentation_path, permissive=True) as mrc:
        membrane = np.asarray(mrc.data) > 0
    if membrane.shape != tomogram.shape:
        raise StampValidationError(f'segmentation shape {membrane.shape} does not match tomogram shape {tomogram.shape} for {job.tomogram_id!r}; they must be the same volume at the same binning')
    # replace membrane voxels with the background median so the class average is not dominated by the membrane slab
    tomogram[membrane] = np.median(tomogram[~membrane])
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(job.output_path), overwrite=True) as out:
        out.set_data(tomogram)
        out.voxel_size = voxel_size
    return job.output_path

# _marker_path: per-tomogram completion marker, written only after its subvolume cache is in place
def _marker_path(cache_dir: Path, tomogram_id: str) -> Path:
    return cache_dir / f'{tomogram_id}.json'

# _write_marker: record which particles were kept or skipped so an aborted run can resume without recomputing
def _write_marker(job: _ExtractTomogramJob, kept: list[Particle], skipped: list[Particle]) -> None:
    marker = {
        'box_voxels': job.box_voxels,
        'subtracted': job.subtracted,
        'kept': [particle.particle_id for particle in kept],
        'skipped': [particle.particle_id for particle in skipped],
    }
    path = _marker_path(job.cache_dir, job.tomogram_id)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(marker))
    os.replace(temporary, path)  # atomic, a partial marker never looks complete

# _read_marker: (kept, skipped, subtracted) from a valid marker for this group, else None
def _read_marker(cache_dir: Path, tomogram_id: str, group: list[Particle], box_voxels: int) -> tuple[list[Particle], list[Particle], bool] | None:
    try:
        marker = json.loads(_marker_path(cache_dir, tomogram_id).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    by_id = {particle.particle_id: particle for particle in group}
    ids = marker.get('kept', []) + marker.get('skipped', [])
    if marker.get('box_voxels') != box_voxels or len(ids) != len(by_id) or set(ids) != set(by_id):
        return None
    if marker['kept'] and not (cache_dir / f'{tomogram_id}.npy').is_file():
        return None
    return [by_id[i] for i in marker['kept']], [by_id[i] for i in marker['skipped']], bool(marker.get('subtracted'))

# _extract_one_tomogram: worker entry point to mmaps one tomogram and extracts subvolumes for its particles
def _extract_one_tomogram(job: _ExtractTomogramJob) -> tuple[list[Particle], list[Particle]]:
    _marker_path(job.cache_dir, job.tomogram_id).unlink(missing_ok=True)  # invalidate any stale marker before rewriting
    kept: list[Particle] = []
    skipped: list[Particle] = []
    subvolumes: list[np.ndarray] = []
    with mrcfile.mmap(job.tomogram_path, permissive=True) as mrc:
        tomogram = mrc.data
        if tomogram.dtype != np.float32:
            tomogram = tomogram.astype(np.float32)
        for particle in job.group:
            if particle.orientation is None or not box_fits_inside(particle.position, tomogram.shape, job.box_voxels):
                skipped.append(particle)
                continue
            subvolumes.append(extract_subvolume(tomogram, particle.position, particle.orientation, job.box_voxels))
            kept.append(particle)
    job.cache_dir.mkdir(parents=True, exist_ok=True)
    if subvolumes:
        final = job.cache_dir / f'{job.tomogram_id}.npy'
        temporary = final.with_name(final.name + '.tmp')
        with open(temporary, 'wb') as handle:
            np.save(handle, np.stack(subvolumes))
        os.replace(temporary, final)  # atomic, a truncated cache never looks complete
    _write_marker(job, kept, skipped)
    return kept, skipped

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

    # subtract membranes to cached mrc files, any leftovers from an aborted run are recomputed
    subtracted_dir = cache_dir / '.subtracted'
    shutil.rmtree(subtracted_dir, ignore_errors=True)
    result_by_tomogram: dict[str, tuple[list[Particle], list[Particle]]] = {}
    subtracted_ids: set[str] = set()
    n_resumed = 0
    subtract_jobs: list[_SubtractTomogramJob] = []
    jobs: list[_ExtractTomogramJob] = []
    for tomogram_id, group in by_tomogram.items():
        path = tomogram_paths.get(tomogram_id)
        if path is None:
            log.warning(f'{tomogram_id}: no matching raw tomogram path, skipping {len(group)} particle(s)')
            skipped.extend(group)
            continue
        resumed = _read_marker(cache_dir, tomogram_id, group, box_voxels)
        if resumed is not None:
            result_by_tomogram[tomogram_id] = (resumed[0], resumed[1])
            if resumed[2]:
                subtracted_ids.add(tomogram_id)
            n_resumed += 1
            continue
        segmentation_path = segmentation_paths.get(tomogram_id)
        if segmentation_path is not None:
            subtract_jobs.append(_SubtractTomogramJob(tomogram_id, path, segmentation_path, subtracted_dir / f'{tomogram_id}.mrc'))
        else:
            jobs.append(_ExtractTomogramJob(tomogram_id, group, path, box_voxels, cache_dir))

    if n_resumed:
        log.info(f'Resuming extraction: {n_resumed}/{len(by_tomogram)} tomogram(s) already extracted')

    def _on_error(job, exc: Exception) -> None:
        if isinstance(exc, StampValidationError):
            log.error(f'{job.tomogram_id}: {exc}')
            skipped.extend(by_tomogram[job.tomogram_id])
        else:
            raise exc

    def _on_subtract_error(job: _SubtractTomogramJob, exc: Exception) -> None:
        if not isinstance(exc, StampValidationError):
            raise exc
        # fall back to the raw tomogram so the particles are still classified, just without subtraction
        log.warning(f'{job.tomogram_id}: {exc}; extracting from the raw tomogram without membrane subtraction')
        jobs.append(_ExtractTomogramJob(job.tomogram_id, by_tomogram[job.tomogram_id], job.tomogram_path, box_voxels, cache_dir))

    def _on_subtracted(job: _SubtractTomogramJob, output_path: Path) -> None:
        jobs.append(_ExtractTomogramJob(job.tomogram_id, by_tomogram[job.tomogram_id], str(output_path), box_voxels, cache_dir, subtracted=True))

    def _on_success(job: _ExtractTomogramJob, result: tuple[list[Particle], list[Particle]]) -> None:
        result_by_tomogram[job.tomogram_id] = result
        if job.subtracted:
            subtracted_ids.add(job.tomogram_id)
            Path(job.tomogram_path).unlink(missing_ok=True)  # free disk as soon as a tomogram is done

    try:
        run_parallel(subtract_jobs, _subtract_one_tomogram, max_workers=n_workers, label='membrane-subtraction', on_success=_on_subtracted, on_error=_on_subtract_error)
        run_parallel(jobs, _extract_one_tomogram, max_workers=n_workers, label='subvolume-extraction', on_success=_on_success, on_error=_on_error)
    finally:
        shutil.rmtree(subtracted_dir, ignore_errors=True)

    index: list[tuple[str, int]] = []
    for tomogram_id in sorted(result_by_tomogram):
        tomogram_kept, tomogram_skipped = result_by_tomogram[tomogram_id]
        kept.extend(tomogram_kept)
        skipped.extend(tomogram_skipped)
        index.extend((tomogram_id, row) for row in range(len(tomogram_kept)))

    log.debug(f'extract_particle_set: {len(kept)} kept, {len(skipped)} skipped')
    n_subtracted = sum(particle.tomogram_id in subtracted_ids for particle in kept)
    parts = [f'{n} ({label})' for n, label in ((n_subtracted, 'membrane-subtracted'), (len(kept) - n_subtracted, 'raw density')) if n]
    log.info(f'Extracted {" + ".join(parts)} subvolumes at box {box_voxels}')
    return Subvolumes(cache_dir, index, box_voxels), kept, skipped
