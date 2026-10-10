'''
STAMP: block tiling of large volumes for streamed processing
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import maximum_filter

# Block: one processing block with block-owned core and read window-derived halo
@dataclass(frozen=True)
class Block:
    index: int
    core_lo: tuple[int, int, int]
    core_hi: tuple[int, int, int]
    read_lo: tuple[int, int, int]
    read_hi: tuple[int, int, int]

# iter_blocks: tile a volume into cores (side: block_size) with a halo; read_lo is rounded down to a multiple of align so binned levels line up
def iter_blocks(shape_zyx: tuple[int, int, int], block_size: int, halo: int, *, align: int = 1) -> list[Block]:
    starts = [range(0, n, block_size) for n in shape_zyx]
    blocks: list[Block] = []
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                core_lo = (z, y, x)
                core_hi = tuple(min(lo + block_size, n) for lo, n in zip(core_lo, shape_zyx))
                read_lo = tuple(max(0, (lo - halo) // align * align) for lo in core_lo)
                read_hi = tuple(min(n, hi + halo) for hi, n in zip(core_hi, shape_zyx))
                blocks.append(Block(len(blocks), core_lo, core_hi, read_lo, read_hi))
    return blocks

# active_blocks: blocks with a core lies within reach_voxels of a segmented voxel, from a bin-8 occupancy grid
def active_blocks(mask: np.ndarray, blocks: list[Block], reach_voxels: float) -> list[Block]:
    cell = 8
    padded = np.pad(mask, [(0, -n % cell) for n in mask.shape])
    occupied = padded.reshape(
        padded.shape[0] // cell, cell, padded.shape[1] // cell, cell, padded.shape[2] // cell, cell
    ).any(axis=(1, 3, 5))
    occupied = maximum_filter(occupied, size=2 * int(np.ceil(reach_voxels / cell)) + 1, mode='constant')
    return [
        block for block in blocks
        if occupied[tuple(slice(lo // cell, (hi - 1) // cell + 1) for lo, hi in zip(block.core_lo, block.core_hi))].any()
    ]

# bin_mean: mean-bin a volume by factor along every axis, cropping the remainder
def bin_mean(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return volume
    z, y, x = (n // factor * factor for n in volume.shape)
    cropped = volume[:z, :y, :x]
    return cropped.reshape(z // factor, factor, y // factor, factor, x // factor, factor).mean(axis=(1, 3, 5), dtype=np.float32)
