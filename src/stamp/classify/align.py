'''
STAMP: reference-based in-plane (azimuthal) alignment of subvolumes
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import rotate

# _ROLL_AXES: box axes spanning the plane perpendicular to the membrane normal
_ROLL_AXES = (0, 1)

# roll_about_normal: rotate a subvolume about its normal by angle_degrees
def roll_about_normal(subvolume: np.ndarray, angle_degrees: float) -> np.ndarray:
    if abs(angle_degrees) < 1e-6:
        return subvolume
    return rotate(subvolume, angle_degrees, axes=_ROLL_AXES, reshape=False, order=1, mode='constant')

# _annulus_mask: True inside the largest cylinder fitting the box, minus a thin core where roll is ill-defined
def _annulus_mask(box_voxels: int, *, min_radius_fraction: float = 0.1) -> np.ndarray:
    half = (box_voxels - 1) / 2.0
    axis = np.arange(box_voxels) - half
    x, y, _z = np.meshgrid(axis, axis, axis, indexing='ij')
    radius = np.sqrt(x * x + y * y)
    return (radius >= min_radius_fraction * half) & (radius <= half)

# _best_angle: angle_degrees maximising masked real-space CC of subvolume against reference
def _best_angle(
    subvolume: np.ndarray,
    reference_masked: np.ndarray,
    mask: np.ndarray,
    angles_degrees: np.ndarray,
) -> float:
    reference_norm = np.linalg.norm(reference_masked) or 1.0
    best_angle, best_score = 0.0, -np.inf
    for angle in angles_degrees:
        rolled = roll_about_normal(subvolume, float(angle))[mask]
        rolled = rolled - rolled.mean()
        score = float(rolled @ reference_masked / ((np.linalg.norm(rolled) or 1.0) * reference_norm))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle

# align_inplace: per-cluster iterative azimuthal alignment; returns {subvolume_row_index: angle_degrees}
def align_inplane(
    subvolumes: np.ndarray,
    cluster_ids: list[str],
    *,
    angular_step_degrees: float = 10.0,
    iterations: int = 3,
) -> dict[int, float]:
    if subvolumes.shape[0] == 0:
        return {}
    box_voxels = subvolumes.shape[-1]
    mask = _annulus_mask(box_voxels)
    angles = np.arange(0.0, 360.0, angular_step_degrees)
    resolved: dict[int, float] = {}
    for cluster_id in sorted(set(cluster_ids)):
        if cluster_id == 'noise':
            continue
        members = [index for index, cid in enumerate(cluster_ids) if cid == cluster_id]
        current = {index: 0.0 for index in members}
        for _ in range(max(iterations, 1)):
            reference = np.mean([roll_about_normal(subvolumes[index], current[index]) for index in members], axis=0)
            reference_masked = reference[mask]
            reference_masked = reference_masked - reference_masked.mean()
            current = {index: _best_angle(subvolumes[index], reference_masked, mask, angles) for index in members}
        resolved.update(current)
    return resolved
