'''
STAMP: class averages from subvolumes
'''

# Import external dependencies
import mrcfile, numpy as np
from pathlib import Path

# compute_class_averages: mean subvolume per cluster
def compute_class_averages(
    subvolumes: np.ndarray, cluster_ids: list[str]
) -> dict[str, tuple[np.ndarray, int]]:
    averages: dict[str, tuple[np.ndarray, int]] = {}
    for cluster_id in sorted(set(cluster_ids)):
        if cluster_id == 'noise':
            continue
        mask = np.array([identifier == cluster_id for identifier in cluster_ids])
        members = subvolumes[mask]
        averages[cluster_id] = (members.mean(axis=0), int(mask.sum()))
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
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(np.transpose(average, (2, 1, 0)).astype(np.float32))
            mrc.voxel_size = voxel_size_angstrom
        written.append(path)
    return written
