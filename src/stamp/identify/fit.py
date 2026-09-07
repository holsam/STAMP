'''
STAMP: rigid real-space fit of a simulated map to a class average
'''

# Import external dependencies
import numpy as np
from itertools import product
from scipy.ndimage import rotate, shift

# Import STAMP schema
from stamp.schemas.particles import IdentificationResult

# _standardise: zero-mean, unit-variance flatten of the masked voxels
def _standardise(values: np.ndarray) -> np.ndarray:
    centred = values - values.mean()
    spread = centred.std()
    return centred / spread if spread > 0 else centred

# _discrete_orients: the 8 axis-aligned orientations left ambiguous after principal-axis alignment
def _discrete_orients(volume: np.ndarray) -> list[np.ndarray]:
    in_plane = [np.rot90(volume, k, axes=(0, 1)) for k in range(4)]
    return in_plane + [np.flip(v, axis=2) for v in in_plane]  # z-flip: which leaflet faces which

# fit_candidate: best masked real-space CC over a bounded rigid search
def fit_candidate(
    class_average: np.ndarray,
    simulated: np.ndarray,
    translation_voxels: int = 3,
    tilt_degrees: float = 12.0,
    tilt_step: float = 6.0,
    *,
    match_contrast: bool = True,
) -> float:
    deviation = np.abs(class_average - class_average.mean())
    mask = deviation > class_average.std()
    if not mask.any():
        mask = np.ones_like(class_average, dtype=bool)
    reference = _standardise(class_average[mask])

    tilts = np.arange(-tilt_degrees, tilt_degrees + 1e-6, tilt_step)
    shifts = range(-translation_voxels, translation_voxels + 1)
    best = -1.0
    for oriented in _discrete_orients(simulated):
        for tilt_yz, tilt_xz in product(tilts, tilts):
            tilted = oriented
            if abs(tilt_yz) > 1e-6:
                tilted = rotate(tilted, float(tilt_yz), axes=(1, 2), reshape=False, order=1, mode='constant')
            if abs(tilt_xz) > 1e-6:
                tilted = rotate(tilted, float(tilt_xz), axes=(0, 2), reshape=False, order=1, mode='constant')
            for dx, dy, dz in product(shifts, shifts, shifts):
                moved = shift(tilted, (dx, dy, dz), order=1, mode='constant', cval=0.0)
                candidate = _standardise(moved[mask])
                score = float(np.dot(reference, candidate) / reference.size)
                best = max(best, abs(score) if match_contrast else score)
    return best

# rank_candidates: ranked IdentificationResult for one class
def rank_candidates(class_id: str, scores: dict[str, float], method: str) -> IdentificationResult:
    if not scores:
        raise ValueError(f'no candidate scores for class {class_id!r}')
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ordered[0]
    gap = float(best_score - ordered[1][1]) if len(ordered) > 1 else None
    return IdentificationResult(
        cluster_id=class_id,
        candidate_protein=best_name,
        fit_score=float(best_score),
        method=method,
        score_gap_to_runner_up=gap,
    )
