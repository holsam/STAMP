'''
STAMP: rotation-invariant spherical-harmonic features of Fourier power spectrum, with each particle's missing wedge as missing data
'''

# Import external dependencies
import numpy as np
from functools import partial
from scipy.ndimage import map_coordinates

# Import internal STAMP objects
from stamp.classify.prepare import normalise, soft_mask
from stamp.classify.spherical import degree_power, fit_matrix, lm_pairs, real_sh_basis, sphere_grid
from stamp.classify.wedge import observed_directions
from stamp.schemas.subvolumes import Subvolumes
from stamp.utils.parallel import run_parallel_ordered

# _MARGIN_DEGREES: angular safety margin inside wedge edge, where interpolation and soft mask leak
_MARGIN_DEGREES = 4.0

# shell_frequencies: n radial shells between low and high frequency in cycles per voxel
def shell_frequencies(n_shells: int, low: float = 0.04, high: float = 0.20) -> np.ndarray:
    return np.linspace(low, high, n_shells)

# sh_power: invariant features (n_shells * n_degrees,) of one axis-aligned box, fit weighted to directions this particle measured, one small ridge solve per particle
def sh_power(
    volume: np.ndarray,
    tilt_max_degrees: float | None,
    tilt_axis_degrees: float,
    *,
    n_shells: int = 8,
    lmax: int = 4,
    ridge: float = 1.0,
    mask_radius_fraction: float = 0.4,
) -> np.ndarray:
    box_voxels = volume.shape[0]
    pairs = lm_pairs(lmax, even_only=True)    # |F|^2 is centrosymmetric
    directions, quadrature = sphere_grid()
    observed = observed_directions(directions, None, tilt_max_degrees, tilt_axis_degrees, margin_degrees=_MARGIN_DEGREES)
    operator = fit_matrix(real_sh_basis(directions, pairs), quadrature * observed, ridge)
    mask = soft_mask(box_voxels, mask_radius_fraction * box_voxels)
    power = np.abs(np.fft.fftn(normalise(volume, mask) * mask)) ** 2
    coordinates = np.stack([shell_frequencies(n_shells)[:, None] * directions[None, :, axis] * box_voxels for axis in range(3)])
    samples = map_coordinates(power, coordinates.reshape(3, -1), order=1, mode='grid-wrap').reshape(n_shells, -1)
    return degree_power(samples @ operator.T, pairs).ravel()

# build_sh_feature_matrix: log-compressed, column-standardised invariant features for every particle of a store of axis-aligned boxes
def build_sh_feature_matrix(subvols: Subvolumes, tilt_max_degrees: float | None, tilt_axis_degrees: float, *, n_workers: int = 1, batch_size: int = 512) -> np.ndarray:
    if len(subvols) == 0:
        return np.empty((0, 0))
    worker = partial(sh_power, tilt_max_degrees=tilt_max_degrees, tilt_axis_degrees=tilt_axis_degrees)
    rows: list[np.ndarray | None] = [None] * len(subvols)
    for indices, batch in subvols.iter_batches(batch_size):
        for index, row in zip(indices, run_parallel_ordered(list(batch), worker, max_workers=n_workers, label='sh-features')):
            rows[index] = row
    logged = np.log(np.stack(rows) + 1e-12)
    spread = logged.std(axis=0, keepdims=True)
    spread[spread == 0.0] = 1.0
    return (logged - logged.mean(axis=0, keepdims=True)) / spread
