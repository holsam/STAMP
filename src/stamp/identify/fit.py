'''
STAMP: rigid real-space fit of a simulated map to a class average
'''

# Import external dependencies
import numpy as np
from itertools import product
from scipy.ndimage import rotate

# Import STAMP schema
from stamp.schemas.particles import IdentificationResult

# _standardise: zero-mean, unit-variance flatten of the masked voxels
def _standardise(values: np.ndarray) -> np.ndarray:
    centred = values - values.mean()
    spread = centred.std()
    return centred / spread if spread > 0 else centred

# fit_candidate: best masked real-space CC over a small rigid search
def fit_candidate(
    class_average: np.ndarray,
    simulated: np.ndarray,
    translation_voxels: int = 3,
    tilt_degrees: float = 15.0,
    tilt_step: float = 5.0
) -> float:
    threshold = class_average.mean() + class_average.std()
    mask = class_average > threshold
    if not mask.any():
        mask = np.ones_like(class_average, dtype=bool)
    reference = _standardise(class_average[mask])

    tilts = np.arange(-tilt_degrees, tilt_degrees + 1e-6, tilt_step)
    shifts = range(-translation_voxels, translation_voxels + 1)
    best = -1.0
    for tilt in tilts:
        tilted = simulated if abs(tilt) < 1e-6 else rotate(
            simulated, float(tilt), axes=(1, 2), reshape=False, order=1, mode='constant'
        )
        for dx, dy, dz in product(shifts, shifts, shifts):
            shifted = np.roll(tilted, (dx, dy, dz), axis=(0, 1, 2))
            candidate = _standardise(shifted[mask])
            score = float(np.dot(reference, candidate) / reference.size)
            if score > best:
                best = score
    return best

# rank_candidates: ranked IdentificationResult for one class
def rank_candidates(class_id: str, scores: dict[str, float], method: str) -> IdentificationResult:
    if not scores:
        raise ValueError(f'no candidate scores for class {class_id}')
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    return IdentificationResult(
        cluster_id=class_id,
        candidate_protein=best_name,
        fit_score=float(best_score),
        method=method,
        score_gap_to_runner_up=float(best_score - runner_up),
    )
