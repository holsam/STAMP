'''
STAMP: rotationally invariant features from subvolumes
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import map_coordinates

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

# cylindrical_resample: resample a canonical subvolume onto a cylindrical (r, phi, z) grid
def cylindrical_resample(
    subvolume: np.ndarray,
    n_radial_bins: int,
    n_azimuthal_samples: int,
) -> np.ndarray:
    box_voxels = subvolume.shape[2]
    half = (box_voxels - 1) / 2.0

    radii = np.linspace(0.5, half, n_radial_bins)
    angles = np.linspace(0.0, 2.0 * np.pi, n_azimuthal_samples, endpoint=False)
    heights = np.arange(box_voxels) - half

    grid_r, grid_phi, grid_z = np.meshgrid(radii, angles, heights, indexing='ij')
    coordinates = np.stack([
            (grid_r * np.cos(grid_phi) + half).ravel(),
            (grid_r * np.sin(grid_phi) + half).ravel(),
            (grid_z + half).ravel(),
    ])

    sampled = map_coordinates(subvolume.astype(np.float32), coordinates, order=1, mode='constant', cval=0.0)
    return sampled.reshape(n_radial_bins, n_azimuthal_samples, box_voxels)

# azimuthal_magnitudes: azimuthal Fourier magnitudes, invariant to in-plane rotation
def azimuthal_magnitudes(
    subvolume: np.ndarray,
    n_radial_bins: int,
    n_azimuthal_samples: int,
    max_mode: int,
) -> np.ndarray:
    if max_mode < 0:
        raise ValueError('max_mode must be non-negative.')
    if n_azimuthal_samples < 2 * (max_mode + 1):
        raise ValueError(f'n_azimuthal_samples ({n_azimuthal_samples}) must be at least 2 * (max_mode + 1) = {2 * (max_mode + 1)} to resolve mode {max_mode} without aliasing')

    cylindrical = cylindrical_resample(subvolume, n_radial_bins, n_azimuthal_samples)
    spectrum = np.fft.rfft(cylindrical, axis=1)
    # normalise by sample count so magnitudes don't scale with sampling
    magnitudes = np.abs(spectrum[:, : max_mode + 1, :]) / n_azimuthal_samples
    # m=0 carries no rotation phase so keep sign
    magnitudes[:, 0, :] = spectrum[:, 0, :].real / n_azimuthal_samples
    # (radial, mode, z) -> (mode, radial, z)
    return np.transpose(magnitudes, (1, 0, 2))

# _standardise_rows: z-score each row, leaving constant rows as zeros rather than NaN
def _standardise_rows(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=1, keepdims=True)
    stds = matrix.std(axis=1, keepdims=True)
    stds[stds == 0.0] = 1.0
    return (matrix - means) / stds

# _standardise_columns: z-score each column across the particle population
def _standardise_columns(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0, keepdims=True)
    stds = matrix.std(axis=0, keepdims=True)
    stds[stds == 0.0] = 1.0
    return (matrix - means) / stds

# build_feature_matrix: feature matrix for a stack of canonical subvolumes
def build_feature_matrix(
    subvolumes: np.ndarray,
    n_radial_bins: int = 12,
    max_azimuthal_mode: int = 4,
    n_azimuthal_samples: int = 64,
    min_radius_fraction: float = 0.25,
) -> np.ndarray:
    if subvolumes.shape[0] == 0:
        return np.empty((0, 0))
    if not 0.0 <= min_radius_fraction < 1.0:
        raise ValueError('min_radius_fraction must be between [0, 1)')

    stacked = np.stack([azimuthal_magnitudes(subvolume, n_radial_bins, n_azimuthal_samples, max_azimuthal_mode) for subvolume in subvolumes])  # (n_particles, n_modes, n_radial_bins, box_voxels)

    n_particles = stacked.shape[0]
    first_outer_bin = int(np.floor(min_radius_fraction * n_radial_bins))

    # Mode 0: all radial bins, standardised per particle
    blocks = [_standardise_rows(stacked[:, 0].reshape(n_particles, -1))]

    # Modes 1+: outer bins only, standardised per column across particle so weak-but-consistent symmetry signal survives into the PCA
    for mode in range(1, max_azimuthal_mode + 1):
        block = stacked[:, mode, first_outer_bin:, :].reshape(n_particles, -1)
        blocks.append(_standardise_columns(block))

    return np.hstack(blocks)
