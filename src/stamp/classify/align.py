'''
STAMP: reference-based in-plane (azimuthal) alignment of subvolumes
'''

# Import external dependencies
import json, numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.ndimage import rotate

# Import internal STAMP objects
from stamp.schemas.subvolumes import Subvolumes
from stamp.utils.checkpoint import atomic_write_json
from stamp.utils.log import log
from stamp.utils.parallel import run_parallel

# _BATCH_SIZE: subvolumes read per chunk in a cluster worker
_BATCH_SIZE = 64

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

# _AlignClusterJob: one cluster's members (lazy view so no subvolume data is held or pickled)
@dataclass(frozen=True)
class _AlignClusterJob:
    cluster_id: str
    member_indices: list[int]
    member_subvolumes: Subvolumes  # local index i <-> member_indices[i]
    mask: np.ndarray
    angles: np.ndarray
    iterations: int

# _align_one_cluster: multiprocessing worker entry point using streamed batches
def _align_one_cluster(job: _AlignClusterJob) -> dict[int, float]:
    n_members = len(job.member_indices)
    current = {local: 0.0 for local in range(n_members)}
    for _ in range(max(job.iterations, 1)):
        total = np.zeros((job.member_subvolumes.box_voxels,) * 3, dtype=np.float64)
        for locals_, batch in job.member_subvolumes.iter_batches(_BATCH_SIZE):
            for local, subvolume in zip(locals_, batch):
                total += roll_about_normal(subvolume, current[local])
        reference_masked = (total / n_members)[job.mask]
        reference_masked = reference_masked - reference_masked.mean()
        updated: dict[int, float] = {}
        for locals_, batch in job.member_subvolumes.iter_batches(_BATCH_SIZE):
            for local, subvolume in zip(locals_, batch):
                updated[local] = _best_angle(subvolume, reference_masked, job.mask, job.angles)
        current = updated
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
        jobs.append(_AlignClusterJob(cluster_id, members, subvols.take(members), mask, angles, iterations))
    if n_resumed:
        log.info(f'Resuming in-plane alignment: {n_resumed} cluster(s) loaded from checkpoint')

    # on_success: checkpoint each cluster as it finishes so an abort only loses in-flight clusters
    def _on_success(job: _AlignClusterJob, cluster_result: dict[int, float]) -> None:
        resolved.update(cluster_result)
        if checkpoint_dir is not None:
            atomic_write_json(checkpoint_dir / f'{job.cluster_id}.json', {'members': job.member_indices, 'angles': [cluster_result[i] for i in job.member_indices]})

    run_parallel(jobs, _align_one_cluster, max_workers=n_workers, label='inplane-align', on_success=_on_success)
    return resolved