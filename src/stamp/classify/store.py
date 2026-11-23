'''
STAMP: working-resolution particles with their orientations, wedges served on demand
'''

# Import external dependencies
import numpy as np, os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.prepare import prepare_particle
from stamp.classify.subvolumes import Subvolumes
from stamp.classify.wedge import box_wedge
from stamp.utils.log import log
from stamp.utils.parallel import run_parallel

# ParticleStore: prepared (normalised, masked, binned) particles on disk plus what is needed to rebuild each particle's wedge
@dataclass(frozen=True)
class ParticleStore:
    volumes: Subvolumes
    orientations: list[tuple[float, float, float, float] | None]
    tilt_max_degrees: float | None
    tilt_axis_degrees: float

    def __len__(self) -> int:
        return len(self.volumes)

    # take: view over a subset of particles, no data copy
    def take(self, indices: Sequence[int]) -> 'ParticleStore':
        return ParticleStore(self.volumes.take(indices), [self.orientations[i] for i in indices], self.tilt_max_degrees, self.tilt_axis_degrees)

    # wedges: box-frame wedges of the given particles
    def wedges(self, indices: Sequence[int]) -> np.ndarray:
        box_voxels = self.volumes.box_voxels
        return np.stack([box_wedge(box_voxels, self.orientations[i], self.tilt_max_degrees, self.tilt_axis_degrees) for i in indices])

    # load: volumes and wedges of the given particles, in the order given
    def load(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        return np.stack([self.volumes[i] for i in indices]), self.wedges(indices)

    # batches: (indices, volumes, wedges) grouped by tomogram for disk-local reads
    def batches(self, batch_size: int = 256) -> Iterator[tuple[list[int], np.ndarray, np.ndarray]]:
        for indices, volumes in self.volumes.iter_batches(batch_size):
            yield indices, volumes, self.wedges(indices)

    # sample: count distinct random particles
    def sample(self, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        return self.load(sorted(rng.choice(len(self), min(count, len(self)), replace=False).tolist()))

# _PrepareJob: one tomogram's source rows and destination
@dataclass(frozen=True)
class _PrepareJob:
    tomogram_id: str
    source: Path
    target: Path
    rows: list[int]
    working_box_voxels: int

# _is_prepared: whether target holds every row at working box
def _is_prepared(job: _PrepareJob) -> bool:
    try:
        return np.load(job.target, mmap_mode='r').shape == (len(job.rows), *(job.working_box_voxels,) * 3)
    except (OSError, ValueError):
        return False

# _prepare_one_tomogram: prepare one tomogram's particles into a partial file, renamed into place when complete
def _prepare_one_tomogram(job: _PrepareJob) -> str:
    if _is_prepared(job):
        return job.tomogram_id
    source = np.load(job.source, mmap_mode='r')
    partial = job.target.with_name(job.target.name + '.partial')    # a crash leaves only this, never a half-written target
    output = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(len(job.rows), *(job.working_box_voxels,) * 3))
    for position, row in enumerate(job.rows):
        output[position] = prepare_particle(np.asarray(source[row]), job.working_box_voxels)
    output.flush()
    del output
    os.replace(partial, job.target)
    return job.tomogram_id

# build_store: prepare every particle of an extracted store at working box and persist result in cache_dir
def build_store(
    subvols: Subvolumes,
    orientations: list[tuple[float, float, float, float] | None],
    working_box_voxels: int,
    tilt_max_degrees: float | None,
    tilt_axis_degrees: float,
    cache_dir: Path,
    *,
    n_workers: int = 1,
) -> ParticleStore:
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_tomogram = subvols.indices_by_tomogram()
    jobs = [_PrepareJob(tomogram_id, subvols.cache_dir / f'{tomogram_id}.npy', cache_dir / f'{tomogram_id}.npy', [subvols.index[i][1] for i in local_indices], working_box_voxels) for tomogram_id, local_indices in by_tomogram.items()]
    n_resumed = sum(_is_prepared(job) for job in jobs)
    if n_resumed:
        log.info(f'Resuming working-box preparation: {n_resumed}/{len(jobs)} tomogram(s) already prepared')
    log.progress(f'Preparing {len(subvols)} particles at working box {working_box_voxels} ({len(jobs)} tomograms)')
    run_parallel(jobs, _prepare_one_tomogram, max_workers=n_workers, label='working-box-preparation')
    new_index: list[tuple[str, int]] = [('', 0)] * len(subvols)
    for tomogram_id, local_indices in by_tomogram.items():
        for position, local_index in enumerate(local_indices):
            new_index[local_index] = (tomogram_id, position)
    return ParticleStore(Subvolumes(cache_dir, new_index, working_box_voxels), orientations, tilt_max_degrees, tilt_axis_degrees)
