'''
STAMP: 2D mocked functions for classify
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import rotate

# Import internal STAMP objects
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

# feature_matrix: stack particle_features over a particle array
def feature_matrix(particles: np.ndarray) -> np.ndarray:
    return np.array([particle_features(p) for p in particles])

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
