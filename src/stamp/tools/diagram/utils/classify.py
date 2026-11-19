'''
STAMP: 2D mocked functions for classify
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import map_coordinates, rotate

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.tools.diagram.utils.identify import standardise

# --- in-plane alignment before averaging ---
# align_one: rigid rotation / shift search aligning one image onto a reference
def align_one(image, reference, step: int = 12, max_shift: int = 2):
    ref = standardise(reference.ravel())
    shifts = range(-max_shift, max_shift + 1)
    best_score, best = -np.inf, image
    for angle in range(0, 360, step):
        rotated = image if angle == 0 else rotate(image, angle, reshape=False, order=1, mode='constant')
        for dy in shifts:
            for dx in shifts:
                candidate = np.roll(rotated, (dy, dx), axis=(0, 1))
                score = float(np.dot(ref, standardise(candidate.ravel())) / ref.size)
                if score > best_score:
                    best_score, best = score, candidate
    return best

# align_stack: align every particle in a stack onto one reference
def align_stack(particles, reference):
    return np.array([align_one(p, reference) for p in particles])

# iterative_average: alternate align-to-reference and re-average for a few passes
def iterative_average(particles, passes: int = 3):
    reference = particles[0]
    aligned = particles
    for _ in range(passes):
        aligned = align_stack(particles, reference)
        reference = aligned.mean(axis=0)
    return aligned, reference

# --- rotation-invariant particle features ---
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
