'''
STAMP: working-resolution particles with their orientations, wedges served on demand
'''

# Import external dependencies
import numpy as np
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.prepare import prepare_particle
from stamp.classify.wedge import box_wedge
from stamp.schemas.subvolumes import Subvolumes

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

# build_store: prepare every particle of an extracted store at working box and persist result in cache_dir
def build_store(
    subvols: Subvolumes,
    orientations: list[tuple[float, float, float, float] | None],
    working_box_voxels: int,
    tilt_max_degrees: float | None,
    tilt_axis_degrees: float,
    cache_dir: Path,
) -> ParticleStore:
    prepared = ((index, prepare_particle(volume, working_box_voxels)) for indices, batch in subvols.iter_batches() for index, volume in zip(indices, batch))
    return ParticleStore(subvols.write_rolled(prepared, cache_dir, box_voxels=working_box_voxels), orientations, tilt_max_degrees, tilt_axis_degrees)
