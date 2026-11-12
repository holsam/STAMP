'''
STAMP: normalisation, soft masking and Fourier binning of subvolumes for alignment
'''

# Import external dependencies
import numpy as np

# radius_grid: distance of every voxel of a box from its centre
def radius_grid(box_voxels: int) -> np.ndarray:
    axis = np.arange(box_voxels) - (box_voxels - 1) / 2.0
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return np.sqrt(x * x + y * y + z * z)

# soft_mask: sphere of 1 inside radius_voxels with a cosine fall to 0 over edge_voxels
def soft_mask(box_voxels: int, radius_voxels: float, edge_voxels: float = 3.0) -> np.ndarray:
    ramp = np.clip((radius_grid(box_voxels) - radius_voxels) / edge_voxels, 0.0, 1.0)
    return (0.5 * (1.0 + np.cos(np.pi * ramp))).astype(np.float32)

# normalise: zero mean and unit variance, with statistics from the voxels where mask > 0.5
def normalise(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inside = volume[mask > 0.5]
    return ((volume - inside.mean()) / (inside.std() + 1e-8)).astype(np.float32)

# working_box: odd edge of a box Fourier-cropped by roughly factor, so centre stays on a voxel
def working_box(box_voxels: int, factor: int) -> int:
    return box_voxels if factor <= 1 else 2 * (box_voxels // (2 * factor)) + 1

# fourier_bin: resample a cube to an odd smaller edge by cropping its Fourier transform, which cannot alias
def fourier_bin(volume: np.ndarray, new_box_voxels: int) -> np.ndarray:
    box_voxels = volume.shape[0]
    if new_box_voxels == box_voxels:
        return volume
    spectrum = np.fft.fftshift(np.fft.fftn(volume))
    low = box_voxels // 2 - new_box_voxels // 2
    cropped = spectrum[low:low + new_box_voxels, low:low + new_box_voxels, low:low + new_box_voxels]
    binned = np.fft.ifftn(np.fft.ifftshift(cropped)).real * (new_box_voxels / box_voxels) ** 3
    return binned.astype(np.float32)

# prepare_particle: normalise inside a soft sphere, mask, then bin to working box
def prepare_particle(volume: np.ndarray, new_box_voxels: int, *, radius_fraction: float = 0.4) -> np.ndarray:
    mask = soft_mask(volume.shape[0], radius_fraction * volume.shape[0])
    return fourier_bin(normalise(volume, mask) * mask, new_box_voxels)
