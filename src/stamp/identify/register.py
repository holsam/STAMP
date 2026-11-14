'''
STAMP: rigid registration of a simulated map to an aligned class average over an orientation prior's search space
'''

# Import external dependencies
import numpy as np
from scipy.spatial.transform import Rotation

# Import internal STAMP objects
from stamp.classify.match import Alignment, align_to_reference
from stamp.classify.prepare import soft_mask
from stamp.classify.search import SearchSpace, search_space
from stamp.schemas.particles import OrientationPrior

# _prepare: soft-masked, mean-removed copy of a map so correlation is not driven by background or edges
def _prepare(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inside = volume[mask > 0.5]
    return ((volume - inside.mean()) * mask).astype(np.float32)

# _local_rotations: n small random rotations up to max_degrees composed onto base
def _local_rotations(base: np.ndarray, n: int, max_degrees: float, rng: np.random.Generator) -> np.ndarray:
    vectors = rng.normal(size=(n, 3))
    vectors *= (np.radians(max_degrees) * rng.random((n, 1)) ** (1 / 3)) / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.einsum('nij,jk->nik', Rotation.from_rotvec(vectors).as_matrix(), base)

# _best_pose: best alignment of average against reference over one search space
def _best_pose(average: np.ndarray, reference: np.ndarray, space: SearchSpace) -> tuple[Alignment, np.ndarray]:
    result = align_to_reference(average[None], np.ones_like(average)[None], reference, space)
    return result, space.rotations[result.rotation_index[0]]

# register_score: best normalised correlation of simulated map with class average over every orientation, a coarse bank then two rounds of local refinement
def register_score(
    average: np.ndarray,
    simulated: np.ndarray,
    prior: OrientationPrior,
    *,
    max_shift_voxels: int = 3,
    n_rotations: int = 1500,
    n_local: int = 200,
    seed: int = 0,
    match_contrast: bool = True,
) -> float:
    box_voxels = average.shape[0]
    mask = soft_mask(box_voxels, 0.45 * box_voxels)
    target = _prepare(average, mask)
    coarse = search_space(OrientationPrior.AXIS if prior is OrientationPrior.NORMAL else OrientationPrior.NONE, box_voxels, n_rotations=n_rotations, shift_range_voxels=max_shift_voxels, seed=seed)   # membrane candidates are slab-oriented, so only the azimuth and the leaflet flip are open
    rng = np.random.default_rng(seed)
    best = -np.inf
    for sign in ((1.0, -1.0) if match_contrast else (1.0,)):
        reference = sign * _prepare(simulated, mask)
        result, rotation = _best_pose(target, reference, coarse)
        score, max_degrees = float(result.cc[0]), 12.0
        for _ in range(2):
            local = _local_rotations(rotation, n_local, max_degrees, rng)
            result, rotation_local = _best_pose(target, reference, SearchSpace(prior, local, np.ones(len(local), dtype=np.int8), coarse.max_shift))
            if result.cc[0] > score:
                score, rotation = float(result.cc[0]), rotation_local
            max_degrees /= 2
        best = max(best, score)
    return best
