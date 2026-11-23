'''
STAMP: lazy index over per-tomogram cached subvolume stacks
'''

# Import external dependencies
import numpy as np
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# _MAX_OPEN_MMAPS: cap on concurrently open per-tomogram mmaps, well under default OS fd limits (macOS ulimit -n defaults to 256)
_MAX_OPEN_MMAPS = 64

# _MmapCache: mutable LRU of open per-tomogram mmaps, held by Subvolumes so the frozen dataclass never reassigns its own mutable state
@dataclass
class _MmapCache:
    entries: OrderedDict[str, np.ndarray] = field(default_factory=OrderedDict)

# Subvolumes: particle -> (tomogram_id, row) index over mmap'd per-tomogram .npy caches
@dataclass(frozen=True)
class Subvolumes:
    cache_dir: Path
    index: list[tuple[str, int]]  # index[i] = (tomogram_id, row) for particle i, in extraction order
    box_voxels: int
    _mmaps: _MmapCache = field(default_factory=_MmapCache, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.index)

    # _mmap_for: open (or reuse) the mmap for one tomogram's cache file, evicting the least-recently-used entry once the cache is full so open file handles stay bounded
    def _mmap_for(self, tomogram_id: str) -> np.ndarray:
        entries = self._mmaps.entries
        cached = entries.get(tomogram_id)
        if cached is not None:
            entries.move_to_end(tomogram_id)
            return cached
        array = np.load(self.cache_dir / f'{tomogram_id}.npy', mmap_mode='r')
        entries[tomogram_id] = array
        if len(entries) > _MAX_OPEN_MMAPS:
            entries.popitem(last=False)  # drop oldest; np.memmap closes its file on garbage collection
        return array

    def __getitem__(self, i: int) -> np.ndarray:
        tomogram_id, row = self.index[i]
        return np.asarray(self._mmap_for(tomogram_id)[row])

    # take: view over a subset of particles by index, no data copy
    def take(self, indices: Sequence[int]) -> 'Subvolumes':
        return Subvolumes(self.cache_dir, [self.index[i] for i in indices], self.box_voxels)

    # indices_by_tomogram: tomogram id -> local indices of its particles, in index order
    def indices_by_tomogram(self) -> dict[str, list[int]]:
        by_tomogram: dict[str, list[int]] = {}
        for local_index, (tomogram_id, _row) in enumerate(self.index):
            by_tomogram.setdefault(tomogram_id, []).append(local_index)
        return by_tomogram

    # iter_batches: yield (global_indices, dense_batch) grouped by tomogram, for disk-local reads
    def iter_batches(self, batch_size: int = 512) -> Iterator[tuple[list[int], np.ndarray]]:
        for tomogram_id, local_indices in self.indices_by_tomogram().items():
            mmap = self._mmap_for(tomogram_id)
            for start in range(0, len(local_indices), batch_size):
                chunk = local_indices[start : start + batch_size]
                rows = [self.index[i][1] for i in chunk]
                yield chunk, np.asarray(mmap[rows])

    # batch_indices: index chunks which would be read by iter_batches, without reading them, so callers can skip batches
    def batch_indices(self, batch_size: int = 512) -> Iterator[list[int]]:
        for local_indices in self.indices_by_tomogram().values():
            for start in range(0, len(local_indices), batch_size):
                yield local_indices[start:start + batch_size]
