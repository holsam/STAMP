'''
STAMP: rotationally invariant features from subvolumes
'''

# Import external dependencies
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from scipy.ndimage import map_coordinates

# Import internal STAMP objects
from stamp.schemas.subvolumes import Subvolumes
from stamp.utils.checkpoint import atomic_save_npz
from stamp.utils.log import get_worker_log_config, init_worker_logging, log
from stamp.utils.parallel import run_parallel_ordered

# FEATURE_NORMALISATIONS: how 1+ modes are made comparable across particles and tomograms
FEATURE_NORMALISATIONS = ('none', 'particle', 'tomogram', 'both')

# _MIN_TOMOGRAM_PARTICLES: smallest tomogram standardised on its own, smaller ones fall back to population statistics
_MIN_TOMOGRAM_PARTICLES = 50

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

# _standardise_columns: z-score each column across the particle population, or within each tomogram of at least _MIN_TOMOGRAM_PARTICLES particles when tomogram_of specified
def _standardise_columns(matrix: np.ndarray, tomogram_of: np.ndarray | None = None) -> np.ndarray:
    # _zscore: standardise block using the statistics of reference
    def _zscore(block: np.ndarray, reference: np.ndarray) -> np.ndarray:
        stds = reference.std(axis=0, keepdims=True)
        stds[stds == 0.0] = 1.0
        return (block - reference.mean(axis=0, keepdims=True)) / stds
    result = _zscore(matrix, matrix)
    if tomogram_of is not None:
        for tomogram in np.unique(tomogram_of):
            members = tomogram_of == tomogram
            if members.sum() >= _MIN_TOMOGRAM_PARTICLES:
                result[members] = _zscore(matrix[members], matrix[members])
    return result

# _normalise_power: divide a particle's 1+ modes by joint L2 norm (so overall non-axisymmetric amplitude doesn't dominate)
def _normalise_power(stacked_modes: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(stacked_modes.reshape(stacked_modes.shape[0], -1), axis=1)
    norms[norms == 0.0] = 1.0
    return stacked_modes / norms[:, None, None, None]

# _load_cached_magnitudes: one tomogram's cached magnitudes, or None when absent or built for different rows
def _load_cached_magnitudes(cache_dir: Path | None, tomogram_id: str, rows: list[int]) -> np.ndarray | None:
    if cache_dir is None:
        return None
    try:
        with np.load(cache_dir / f'{tomogram_id}.npz') as cached:
            if np.array_equal(cached['rows'], rows):
                return cached['magnitudes']
    except (OSError, ValueError, KeyError):
        pass
    return None

# build_feature_matrix: feature matrix for a Subvolumes instance, read in per-tomogram batches, per-tomogram magnitudes cached in cache_dir when given
def build_feature_matrix(
    subvols: Subvolumes,
    n_radial_bins: int = 12,
    max_azimuthal_mode: int = 4,
    n_azimuthal_samples: int = 64,
    min_radius_fraction: float = 0.25,
    n_workers: int = 1,
    *,
    batch_size: int = 512,
    cache_dir: Path | None = None,
    normalisation: str = 'both',
) -> np.ndarray:
    log.debug(f'Building features for {len(subvols)} subvolumes, max_mode={max_azimuthal_mode}')
    if len(subvols) == 0:
        return np.empty((0, 0))
    if not 0.0 <= min_radius_fraction < 1.0:
        raise ValueError('min_radius_fraction must be between [0, 1)')
    if normalisation not in FEATURE_NORMALISATIONS:
        raise ValueError(f'normalisation must be one of {FEATURE_NORMALISATIONS}, got {normalisation!r}')

    worker = partial(azimuthal_magnitudes, n_radial_bins=n_radial_bins, n_azimuthal_samples=n_azimuthal_samples, max_mode=max_azimuthal_mode)
    magnitudes: list[np.ndarray | None] = [None] * len(subvols)
    total_particles = len(subvols)
    # create process pool for feature extraction to be reused across batches
    pool_context = (
        ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker_logging, initargs=get_worker_log_config())
        if n_workers > 1
        else nullcontext()
    )
    by_tomogram: dict[str, list[int]] = {}
    for local_index, (tomogram_id, _row) in enumerate(subvols.index):
        by_tomogram.setdefault(tomogram_id, []).append(local_index)
    with pool_context as pool:
        completed = 0
        n_resumed = 0
        for tomogram_id, local_indices in by_tomogram.items():
            rows = [subvols.index[i][1] for i in local_indices]
            tomogram_magnitudes = _load_cached_magnitudes(cache_dir, tomogram_id, rows)
            if tomogram_magnitudes is None:
                results: list[np.ndarray] = []
                for _indices, batch in subvols.take(local_indices).iter_batches(batch_size):
                    results.extend(run_parallel_ordered(
                        list(batch),
                        worker,
                        max_workers=n_workers,
                        label='feature-extraction',
                        pool=pool,
                        progress_total=total_particles,
                        progress_offset=completed + len(results),
                    ))
                tomogram_magnitudes = np.stack(results)
                if cache_dir is not None:
                    atomic_save_npz(cache_dir / f'{tomogram_id}.npz', rows=np.array(rows), magnitudes=tomogram_magnitudes)
            else:
                n_resumed += 1
                log.debug(f'{tomogram_id}: features loaded from checkpoint')
            completed += len(local_indices)
            for local_index, result in zip(local_indices, tomogram_magnitudes):
                magnitudes[local_index] = result
    if n_resumed:
        log.info(f'Resuming feature extraction: {n_resumed}/{len(by_tomogram)} tomogram(s) loaded from checkpoint')
    stacked = np.stack(magnitudes)  # (n_particles, n_modes, n_radial_bins, box_voxels)

    n_particles = stacked.shape[0]
    first_outer_bin = int(np.floor(min_radius_fraction * n_radial_bins))

    # Mode 0: all radial bins, standardised per particle
    blocks = [_standardise_rows(stacked[:, 0].reshape(n_particles, -1))]

    # Modes 1+: outer bins only, standardised per column so weak-but-consistent symmetry signal survives into the PCA
    # 'particle' first divides each particle by its own total 1+ mode power, 'tomogram' standardises columns within each tomogram
    tomogram_of = np.empty(n_particles, dtype=np.int64)
    for tomogram_number, local_indices in enumerate(by_tomogram.values()):
        tomogram_of[local_indices] = tomogram_number
    outer = stacked[:, 1:, first_outer_bin:, :]
    if normalisation in ('particle', 'both') and outer.shape[1]:
        outer = _normalise_power(outer)
    for mode in range(outer.shape[1]):
        block = outer[:, mode].reshape(n_particles, -1)
        blocks.append(_standardise_columns(block, tomogram_of if normalisation in ('tomogram', 'both') else None))

    features = np.hstack(blocks)
    features[~stacked.reshape(n_particles, -1).any(axis=1)] = 0.0  # empty subvolumes stay all-zero (column standardisation gives non-zero rows that aren't recognised as degenerate)
    return features