'''
STAMP: phase randomisation
'''

# Import external dependencies
import numpy as np

# phase_randomise: keep Fourier amplitudes and replace phases by those of white noise, outside the voxels flagged in keep (Hermitian noise spectrum)
def phase_randomise(volume: np.ndarray, rng: np.random.Generator, *, keep: np.ndarray | None = None) -> np.ndarray:
    transform = np.fft.fftn(volume)
    noise_phase = np.exp(1j * np.angle(np.fft.fftn(rng.normal(size=volume.shape))))
    randomised = np.abs(transform) * noise_phase
    if keep is not None:
        randomised[keep] = transform[keep]
    return np.fft.ifftn(randomised).real.astype(np.float32)
