'''
STAMP: 2D mocked functions for refine
'''

# Import external dependencies
import numpy as np

# fourier_ring_correlation: calculate FRC
def fourier_ring_correlation(map_a, map_b):
    fft_a = np.fft.fftshift(np.fft.fft2(map_a))
    fft_b = np.fft.fftshift(np.fft.fft2(map_b))
    ny, nx = map_a.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    radius = np.hypot(yy - ny // 2, xx - nx // 2).astype(int)
    n_shells = radius.max()
    frequencies = np.arange(n_shells) / nx
    correlation = np.zeros(n_shells)
    for index in range(n_shells):
        selection = radius == index
        cross = np.sum(fft_a[selection] * np.conj(fft_b[selection]))
        power = np.sqrt(np.sum(np.abs(fft_a[selection]) ** 2) * np.sum(np.abs(fft_b[selection]) ** 2))
        correlation[index] = np.real(cross / power) if power > 0 else 0.0
    return frequencies, correlation

# resolution_at_threshold: calculate resolution
def resolution_at_threshold(frequencies, correlation, pixel_size, threshold: float = 0.143) -> float:
    below = np.where(correlation < threshold)[0]
    if len(below) == 0 or below[0] == 0:
        return float('inf')
    freq = frequencies[below[0]]
    return pixel_size / freq if freq > 0 else float('inf')
