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

    # write_rolled: persist (global_index, subvolume) pairs to a new per-tomogram cache via preallocated memmaps, return the resulting store
    def write_rolled(self, rolled: Iterator[tuple[int, np.ndarray]], new_cache_dir: Path, *, box_voxels: int | None = None) -> 'Subvolumes':
        new_cache_dir.mkdir(parents=True, exist_ok=True)
        row_counts: dict[str, int] = {}
        new_index: list[tuple[str, int]] = []
        for tomogram_id, _row in self.index:
            new_index.append((tomogram_id, row_counts.get(tomogram_id, 0)))
            row_counts[tomogram_id] = row_counts.get(tomogram_id, 0) + 1
        new_box_voxels = box_voxels or self.box_voxels
        shape = (new_box_voxels,) * 3
        outputs: dict[str, np.ndarray] = {}
        written = 0
        for global_index, subvolume in rolled:
            tomogram_id, row = new_index[global_index]
            if tomogram_id not in outputs:
                if len(outputs) >= _MAX_OPEN_MMAPS:  # flush and close the oldest to bound open handles
                    outputs.pop(next(iter(outputs))).flush()
                outputs[tomogram_id] = np.lib.format.open_memmap(new_cache_dir / f'{tomogram_id}.npy', mode='r+' if (new_cache_dir / f'{tomogram_id}.npy').exists() else 'w+', dtype=np.float32, shape=(row_counts[tomogram_id], *shape))
            outputs[tomogram_id][row] = subvolume
            written += 1
        for output in outputs.values():
            output.flush()
        assert written == len(self.index)  # every particle must be rolled exactly once
        return Subvolumes(new_cache_dir, new_index, new_box_voxels)
