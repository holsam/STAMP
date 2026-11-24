'''
STAMP: 2D mocked functions for classify
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import map_coordinates

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior

# --- rotation-invariant particle features ---
# class_label: class id as classify writes it, prefixed with the orientation prior (normal-c00)
def class_label(prior: OrientationPrior, cluster_id: int) -> str:
    return f'{prior.value}-c{cluster_id:02d}'

# radial_profile: mean intensity per radial bin, invariant to in-plane rotation
def radial_profile(image: np.ndarray, n_bins: int = 24) -> np.ndarray:
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    radius = np.hypot(yy - (ny - 1) / 2.0, xx - (nx - 1) / 2.0)
    edges = np.linspace(0, radius.max(), n_bins + 1)
    bin_index = np.clip(np.digitize(radius, edges) - 1, 0, n_bins - 1)
    sums = np.bincount(bin_index.ravel(), weights=image.ravel(), minlength=n_bins)
    counts = np.bincount(bin_index.ravel(), minlength=n_bins)
    return sums / np.maximum(counts, 1)

# annular_fourier_features: log mean power per annulus of the amplitude spectrum
def annular_fourier_features(image: np.ndarray, n_bins: int = 16) -> np.ndarray:
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    radius = np.hypot(yy - ny // 2, xx - nx // 2)
    edges = np.linspace(0, radius.max(), n_bins + 1)
    bin_index = np.clip(np.digitize(radius, edges) - 1, 0, n_bins - 1)
    sums = np.bincount(bin_index.ravel(), weights=spectrum.ravel(), minlength=n_bins)
    counts = np.bincount(bin_index.ravel(), minlength=n_bins)
    return np.log1p(sums / np.maximum(counts, 1))

# particle_features: radial profile concatenated with annular Fourier features
def particle_features(image: np.ndarray) -> np.ndarray:
    return np.concatenate([radial_profile(image), annular_fourier_features(image)])

# axial_features: low-frequency log amplitude spectrum, invariant to axial shift and to 180 degree polarity flip of axis prior
def axial_features(image: np.ndarray, half_width: int = 8) -> np.ndarray:
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    centre = image.shape[0] // 2
    return np.log1p(spectrum[centre - half_width:centre + half_width + 1, centre - half_width:centre + half_width + 1]).ravel()

# polar_mode_features: magnitude of each circular-harmonic mode on rings of increasing radius, invariant to any in-plane rotation
def polar_mode_features(image: np.ndarray, n_radii: int = 10, max_mode: int = 6, n_angles: int = 64) -> np.ndarray:
    centre = (image.shape[0] - 1) / 2.0
    radii = np.linspace(0.05, 0.45, n_radii) * image.shape[0]
    theta = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    rings = map_coordinates(image, [centre + radii[:, None] * np.sin(theta), centre + radii[:, None] * np.cos(theta)], order=1)
    return np.abs(np.fft.rfft(rings, axis=1))[:, :max_mode + 1].ravel()

# feature_matrix: per-particle features that are invariant to what prior leaves unknown
def feature_matrix(particles: np.ndarray, prior: OrientationPrior = OrientationPrior.NORMAL) -> np.ndarray:
    extract = {OrientationPrior.NORMAL: particle_features, OrientationPrior.AXIS: axial_features, OrientationPrior.NONE: polar_mode_features}[prior]
    return np.array([extract(p) for p in particles])

# standardise_columns: zero-mean, unit-variance each feature column
def standardise_columns(features: np.ndarray) -> np.ndarray:
    return (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)

# pca_2d: project standardised features onto their first two principal components
def pca_2d(features: np.ndarray) -> np.ndarray:
    centred = standardise_columns(features)
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T

# --- post-averaging noise model ---
# add_averaging_noise: add Gaussian noise after aligning and averaging particles
def add_averaging_noise(image: np.ndarray, per_particle_snr: float, n_averaged: int, rng) -> np.ndarray:
    signal_rms = image.std() or 1.0
    sigma = signal_rms / per_particle_snr / np.sqrt(max(n_averaged, 1))
    return image + rng.normal(0.0, sigma, size=image.shape)
