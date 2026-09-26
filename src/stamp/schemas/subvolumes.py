'''
STAMP: lazy index over per-tomogram cached subvolume stacks
'''

# Import external dependencies
import numpy as np
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Subvolumes: particle -> (tomogram_id, row) index over mmap'd per-tomogram .npy caches
@dataclass(frozen=True)
class Subvolumes:
    cache_dir: Path
    index: list[tuple[str, int]]  # index[i] = (tomogram_id, row) for particle i, in extraction order
    box_voxels: int
    _mmaps: dict[str, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.index)

    # _mmap_for: open (or reuse) the mmap for one tomogram's cache file
    def _mmap_for(self, tomogram_id: str) -> np.ndarray:
        cached = self._mmaps.get(tomogram_id)
        if cached is not None:
            return cached
        array = np.load(self.cache_dir / f'{tomogram_id}.npy', mmap_mode='r')
        self._mmaps[tomogram_id] = array  # mutating the dict's contents, not the frozen field itself
        return array

    def __getitem__(self, i: int) -> np.ndarray:
        tomogram_id, row = self.index[i]
        return np.asarray(self._mmap_for(tomogram_id)[row])

    # take: view over a subset of particles by index, no data copy
    def take(self, indices: Sequence[int]) -> 'Subvolumes':
        return Subvolumes(self.cache_dir, [self.index[i] for i in indices], self.box_voxels)

    # iter_batches: yield (global_indices, dense_batch) grouped by tomogram, for disk-local reads
    def iter_batches(self, batch_size: int = 512) -> Iterator[tuple[list[int], np.ndarray]]:
        by_tomogram: dict[str, list[int]] = {}
        for local_index, (tomogram_id, _row) in enumerate(self.index):
            by_tomogram.setdefault(tomogram_id, []).append(local_index)
        for tomogram_id, local_indices in by_tomogram.items():
            mmap = self._mmap_for(tomogram_id)
            for start in range(0, len(local_indices), batch_size):
                chunk = local_indices[start : start + batch_size]
                rows = [self.index[i][1] for i in chunk]
                yield chunk, np.asarray(mmap[rows])

    # write_rolled: persist (global_index, subvolume) pairs to a new per-tomogram cache, return the resulting store
    def write_rolled(self, rolled: Iterator[tuple[int, np.ndarray]], new_cache_dir: Path) -> 'Subvolumes':
        new_cache_dir.mkdir(parents=True, exist_ok=True)
        buffers: dict[str, list[np.ndarray]] = {}
        order: dict[str, list[int]] = {}
        for global_index, subvolume in rolled:
            tomogram_id, _row = self.index[global_index]
            buffers.setdefault(tomogram_id, []).append(subvolume)
            order.setdefault(tomogram_id, []).append(global_index)
        new_index: list[tuple[str, int] | None] = [None] * len(self.index)
        for tomogram_id, subvolumes in buffers.items():
            np.save(new_cache_dir / f'{tomogram_id}.npy', np.stack(subvolumes))
            for row, global_index in enumerate(order[tomogram_id]):
                new_index[global_index] = (tomogram_id, row)
        assert all(entry is not None for entry in new_index)  # every particle must be rolled exactly once
        return Subvolumes(new_cache_dir, new_index, self.box_voxels)  # type: ignore[arg-type]
