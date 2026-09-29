'''
STAMP: reference-based in-plane (azimuthal) alignment of subvolumes
'''

# Import external dependencies
import numpy as np
import json
from dataclasses import dataclass
from pathlib import Path
from scipy.ndimage import rotate

# Import internal STAMP objects
from stamp.schemas.subvolumes import Subvolumes
from stamp.utils.checkpoint import atomic_write_json
from stamp.utils.log import log
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
    cluster_id: str
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

# _load_cluster_checkpoint: saved angles for a cluster, or None when absent or computed for different members
def _load_cluster_checkpoint(checkpoint_dir: Path | None, cluster_id: str, members: list[int]) -> dict[int, float] | None:
    if checkpoint_dir is None:
        return None
    try:
        saved = json.loads((checkpoint_dir / f'{cluster_id}.json').read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if saved.get('members') != members:
        return None
    return dict(zip(members, saved['angles']))

# align_inplane: per-cluster iterative azimuthal alignment, each finished cluster checkpointed in checkpoint_dir when given; returns {particle_index: angle_degrees}
def align_inplane(
    subvols: Subvolumes,
    cluster_ids: list[str],
    *,
    angular_step_degrees: float = 10.0,
    iterations: int = 3,
    n_workers: int = 1,
    checkpoint_dir: Path | None = None,
) -> dict[int, float]:
    if len(subvols) == 0:
        return {}
    box_voxels = subvols.box_voxels
    mask = _annulus_mask(box_voxels)
    angles = np.arange(0.0, 360.0, angular_step_degrees)
    jobs = []
    resolved: dict[int, float] = {}
    n_resumed = 0
    for cluster_id in sorted(set(cluster_ids)):
        if cluster_id == 'noise':
            continue
        members = [index for index, cid in enumerate(cluster_ids) if cid == cluster_id]
        saved = _load_cluster_checkpoint(checkpoint_dir, cluster_id, members)
        if saved is not None:
            resolved.update(saved)
            n_resumed += 1
            continue
        member_subvolumes = np.stack([subvols[i] for i in members])  # bounded to one cluster's members, not the whole dataset
        jobs.append(_AlignClusterJob(cluster_id, members, member_subvolumes, mask, angles, iterations))
    if n_resumed:
        log.info(f'Resuming in-plane alignment: {n_resumed} cluster(s) loaded from checkpoint')

    # on_success: checkpoint each cluster as it finishes so an abort only loses in-flight clusters
    def _on_success(job: _AlignClusterJob, cluster_result: dict[int, float]) -> None:
        resolved.update(cluster_result)
        if checkpoint_dir is not None:
            atomic_write_json(checkpoint_dir / f'{job.cluster_id}.json', {'members': job.member_indices, 'angles': [cluster_result[i] for i in job.member_indices]})

    run_parallel(jobs, _align_one_cluster, max_workers=n_workers, label='inplane-align', on_success=_on_success)
    return resolved