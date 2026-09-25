'''
STAMP: reference-based in-plane (azimuthal) alignment of subvolumes
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import rotate

# Import internal STAMP objects
from stamp.schemas.subvolumes import Subvolumes
from stamp.utils.parallel import run_parallel

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

# _AlignClusterJob: one cluster's subvolumes
@dataclass(frozen=True)
class _AlignClusterJob:
    member_indices: list[int]
    member_subvolumes: np.ndarray
    mask: np.ndarray
    angles: np.ndarray
    iterations: int

# _align_one_cluster: multiprocessing worker entry point
def _align_one_cluster(job: _AlignClusterJob) -> dict[int, float]:
    current = {local: 0.0 for local in range(len(job.member_indices))}
    for _ in range(max(job.iterations, 1)):
        reference = np.mean([roll_about_normal(job.member_subvolumes[local], angle) for local, angle in current.items()], axis=0)
        reference_masked = reference[job.mask]
        reference_masked = reference_masked - reference_masked.mean()
        current = {local: _best_angle(job.member_subvolumes[local], reference_masked, job.mask, job.angles) for local in current}
    return {job.member_indices[local]: angle for local, angle in current.items()}

# align_inplane: per-cluster iterative azimuthal alignment; returns {particle_index: angle_degrees}
def align_inplane(
    subvols: Subvolumes,
    cluster_ids: list[str],
    *,
    angular_step_degrees: float = 10.0,
    iterations: int = 3,
    n_workers: int = 1,
) -> dict[int, float]:
    if len(subvols) == 0:
        return {}
    box_voxels = subvols.box_voxels
    mask = _annulus_mask(box_voxels)
    angles = np.arange(0.0, 360.0, angular_step_degrees)
    jobs = []
    for cluster_id in sorted(set(cluster_ids)):
        if cluster_id == 'noise':
            continue
        members = [index for index, cid in enumerate(cluster_ids) if cid == cluster_id]
        member_subvolumes = np.stack([subvols[i] for i in members])  # bounded to one cluster's members, not the whole dataset
        jobs.append(_AlignClusterJob(members, member_subvolumes, mask, angles, iterations))
    resolved: dict[int, float] = {}
    for cluster_result in run_parallel(jobs, _align_one_cluster, max_workers=n_workers, label='inplane-align'):
        resolved.update(cluster_result)
    return resolved