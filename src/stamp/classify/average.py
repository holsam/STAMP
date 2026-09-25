'''
STAMP: class averages from subvolumes
'''

# Import external dependencies
import mrcfile, numpy as np
from pathlib import Path

# Import internal STAMP objects
from stamp.schemas.subvolumes import Subvolumes
from stamp.utils.log import log

# compute_class_averages: mean subvolume per cluster, streamed in batches
def compute_class_averages(
    subvols: Subvolumes,
    cluster_ids: list[str],
    *,
    batch_size: int = 512,
) -> dict[str, tuple[np.ndarray, int]]:
    averages: dict[str, tuple[np.ndarray, int]] = {}
    for cluster_id in sorted(set(cluster_ids)):
        if cluster_id == 'noise':
            continue
        members = [index for index, cid in enumerate(cluster_ids) if cid == cluster_id]
        member_subvols = subvols.take(members)
        total = np.zeros((subvols.box_voxels,) * 3, dtype=np.float64)
        count = 0
        for _indices, batch in member_subvols.iter_batches(batch_size):
            total += batch.sum(axis=0, dtype=np.float64)
            count += batch.shape[0]
        averages[cluster_id] = ((total / count).astype(np.float32), count)
    return averages

# write_class_averages: write each class average as an MRC with its voxel size set
def write_class_averages(
    averages: dict[str, tuple[np.ndarray, int]],
    output_dir: Path,
    voxel_size_angstrom: float,
    half_set_label: str | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cluster_id, (average, count) in averages.items():
        suffix = f'_half{half_set_label}' if half_set_label else ''
        path = output_dir / f'{cluster_id}{suffix}_n{count}.mrc'
        log.debug(f'Writing class average {cluster_id} (half-set: {half_set_label})')
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(np.transpose(average, (2, 1, 0)).astype(np.float32))
            mrc.voxel_size = voxel_size_angstrom
        written.append(path)
    return written
