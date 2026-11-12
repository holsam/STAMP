'''
STAMP: alignment search space for each orientation prior
'''

# Import external dependencies
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

# Import internal STAMP objects
from stamp.classify.frames import azimuth_matrix, flip_matrix, rotate_density
from stamp.schemas.particles import OrientationPrior

# SearchSpace: candidate rotations (with the polarity each one carries) and the shift window searched around each
@dataclass(frozen=True)
class SearchSpace:
    prior: OrientationPrior
    rotations: np.ndarray   # (n, 3, 3), reference moved by rotations[i] gives candidate i
    polarity: np.ndarray    # (n,), -1 for candidates containing the flip, +1 otherwise
    max_shift: tuple[int, int, int]     # largest shift searched along x, y, z in working voxels

    def __len__(self) -> int:
        return len(self.rotations)

# search_space: rotations and shift window for a prior at working box size box_voxels
def search_space(
    prior: OrientationPrior,
    box_voxels: int,
    *,
    angular_step_degrees: float = 10.0,
    n_rotations: int = 3000,
    shift_range_voxels: int = 3,
    seed: int = 0,
) -> SearchSpace:
    lateral = (shift_range_voxels,) * 2
    if prior is OrientationPrior.NORMAL:
        angles = np.arange(0.0, 360.0, angular_step_degrees)
        rotations = np.stack([azimuth_matrix(a) for a in angles])
        return SearchSpace(prior, rotations, np.ones(len(rotations), dtype=np.int8), (*lateral, shift_range_voxels))
    if prior is OrientationPrior.AXIS:
        angles = np.arange(0.0, 360.0, angular_step_degrees)
        upright = np.stack([azimuth_matrix(a) for a in angles])
        rotations = np.concatenate([upright, upright @ flip_matrix()])
        polarity = np.concatenate([np.ones(len(angles)), -np.ones(len(angles))]).astype(np.int8)
        return SearchSpace(prior, rotations, polarity, (*lateral, max(shift_range_voxels, box_voxels // 4)))   # a quarter box either side covers a few helical repeats
    random = Rotation.random(n_rotations - 1, random_state=seed).as_matrix()
    rotations = np.concatenate([np.eye(3)[None], random])   # identity first so a pre-aligned set can win
    return SearchSpace(prior, rotations, np.ones(len(rotations), dtype=np.int8), (*lateral, shift_range_voxels))

# rotated_bank: the reference moved by every candidate rotation, flattened to (n, voxels)
def rotated_bank(reference: np.ndarray, rotations: np.ndarray, *, threads: int = 1) -> np.ndarray:
    if threads <= 1:
        bank = [rotate_density(reference, rotation) for rotation in rotations]
    else:
        with ThreadPoolExecutor(threads) as pool:   # scipy.ndimage releases the GIL
            bank = list(pool.map(lambda rotation: rotate_density(reference, rotation), rotations))
    return np.stack(bank).reshape(len(rotations), -1).astype(np.float32)
