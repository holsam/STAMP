'''
STAMP: half-split ceilings and reference merging
'''

# Import external dependencies
import numpy as np
from collections.abc import Iterable

# Import internal STAMP objects
from stamp.classify.average import WeightedAverage
from stamp.classify.match import align_to_reference
from stamp.classify.search import SearchSpace

# pairwise_similarity: symmetric matrix of best-pose normalised correlation between references, and pose mapping reference j onto reference i (references are coverage-corrected so no wedge)
def pairwise_similarity(references: list[np.ndarray], space: SearchSpace, *, threads: int = 1) -> tuple[np.ndarray, list[list[tuple[np.ndarray, np.ndarray]]]]:
    n = len(references)
    stack = np.stack(references)
    full = np.ones_like(stack)
    similarity = np.eye(n)
    poses: list[list[tuple[np.ndarray, np.ndarray]]] = [[(np.eye(3), np.zeros(3)) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        result = align_to_reference(stack, full, references[i], space, threads=threads)
        similarity[i] = result.cc
        for j in range(n):
            poses[i][j] = (space.rotations[result.rotation_index[j]], result.shift[j].astype(float))
    return np.maximum(similarity, similarity.T), poses

# merge_groups: connected components of the graph whose edges are similarities at or above threshold (scalar or per-pair matrix)
def merge_groups(similarity: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    n = len(similarity)
    group = np.arange(n)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= (threshold[i, j] if np.ndim(threshold) else threshold):
                group[group == group[j]] = group[i]
    return np.unique(group, return_inverse=True)[1]

# pair_thresholds: merge threshold per pair (fraction of the higher of the two self-agreement ceilings) so an inconsistent reference cannot lower the bar
def pair_thresholds(ceilings: np.ndarray, fraction: float) -> np.ndarray:
    return fraction * np.maximum(ceilings[:, None], ceilings[None, :])

# half_split_ceilings: per-label agreement between the averages of two random halves of its members, streamed from batches of (indices, volumes, wedges)
def half_split_ceilings(
    batches: Iterable[tuple[list[int], np.ndarray, np.ndarray]],
    labels: np.ndarray,
    rotations: np.ndarray,
    shifts: np.ndarray,
    space: SearchSpace,
    n_labels: int,
    box_voxels: int,
    *,
    min_members: int = 8,
    seed: int = 0,
) -> np.ndarray:
    side = np.random.default_rng(seed).integers(0, 2, len(labels))
    sums = {(k, s): WeightedAverage(box_voxels) for k in range(n_labels) for s in (0, 1)}
    for indices, volumes, wedges in batches:
        indices = np.array(indices)
        for (k, s), accumulator in sums.items():
            chosen = (labels[indices] == k) & (side[indices] == s)
            if chosen.any():
                accumulator.add(volumes[chosen], wedges[chosen], rotations[indices[chosen]], shifts[indices[chosen]])
    out = np.full(n_labels, np.nan)
    for k in range(n_labels):
        if sums[k, 0].count + sums[k, 1].count >= min_members and sums[k, 0].count and sums[k, 1].count:
            out[k] = pairwise_similarity([sums[k, 0].result(), sums[k, 1].result()], space)[0][0, 1]
    return np.where(np.isnan(out), np.nanmedian(out) if not np.isnan(out).all() else 1.0, out)
