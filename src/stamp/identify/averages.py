'''
STAMP: class average loading grouped by orientation prior
'''

# Import external dependencies
import re, numpy as np
from collections import defaultdict
from pathlib import Path

# Import STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.utils.errors import StampPipelineError
from stamp.utils.mrc import glob_mrc, read_class_average_mrc

# _CLASS_ID: leading {prior}-cNN token of a class-average filename
_CLASS_ID = re.compile(rf'^((?:{"|".join(prior.value for prior in OrientationPrior)})-c\d+)')

# load_class_averages: {class_id: (mean_volume, voxel_size)} merged across half-set MRCs
def load_class_averages(directory: Path, *, allow_empty: bool = False) -> dict[str, tuple[np.ndarray, float]]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    voxel_sizes: dict[str, float] = {}
    for path in (glob_mrc(directory) if directory.is_dir() else []):
        match = _CLASS_ID.match(path.stem)
        if not match:
            continue
        volume, voxel_size = read_class_average_mrc(path)
        grouped[match.group(1)].append(volume)
        voxel_sizes[match.group(1)] = voxel_size
    if not grouped:
        if allow_empty:
            return {}
        raise StampPipelineError(f'No {{prior}}-cNN class averages in {directory}')
    return {cid: (np.mean(volumes, axis=0), voxel_sizes[cid]) for cid, volumes in grouped.items()}
