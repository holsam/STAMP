'''
STAMP: rotationally invariant features from subvolumes
'''

# Import external dependencies
import numpy as np

# cylindrical_bins: precompute per-voxel radial bin index and validity mask
def cylindrical_bins(box_voxels: int, n_radial_bins: int) -> tuple[np.ndarray, np.ndarray]:
    half = (box_voxels - 1) / 2.0
    axis = np.arange(box_voxels) - half
    x, y, _z = np.meshgrid(axis, axis, axis, indexing='ij')
    radius = np.sqrt(x * x + y * y)
    max_radius = half
    valid = radius <= max_radius
    bin_index = np.floor(radius / max_radius * n_radial_bins).astype(np.int64)
    bin_index = np.clip(bin_index, 0, n_radial_bins - 1)
    return bin_index, valid

# rotational_average: average a canonical subvolume around its z axis
def rotational_average(
    subvolume: np.ndarray,
    bin_index: np.ndarray,
    valid: np.ndarray,
    n_radial_bins: int
) -> np.ndarray:
    box_voxels = subvolume.shape[2]
    flat_index = (bin_index * box_voxels + np.arange(box_voxels)[None, None, :])
    counts = np.bincount(
        flat_index[valid].ravel(), minlength=n_radial_bins * box_voxels
    ).astype(np.float64)
    sums = np.bincount(
        flat_index[valid].ravel(),
        weights=subvolume[valid].ravel(),
        minlength=n_radial_bins * box_voxels,
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        averaged = np.where(counts > 0, sums / counts, 0.0)
    return averaged.reshape(n_radial_bins, box_voxels)

# build_feature_matrix: feature matrix (n_particles, n_radial_bins * box_voxels)
def build_feature_matrix(
    subvolumes: np.ndarray,
    n_radial_bins: int = 12
) -> np.ndarray:
    if subvolumes.shape[0] == 0:
        return np.empty((0, n_radial_bins * subvolumes.shape[-1]))
    box_voxels = subvolumes.shape[-1]
    bin_index, valid = cylindrical_bins(box_voxels, n_radial_bins)
    features = np.stack(
        [
            rotational_average(subvolume, bin_index, valid, n_radial_bins).ravel()
            for subvolume in subvolumes
        ]
    )
    means = features.mean(axis=1, keepdims=True)
    stds = features.std(axis=1, keepdims=True)
    stds[stds == 0.0] = 1.0
    return (features - means) / stds
