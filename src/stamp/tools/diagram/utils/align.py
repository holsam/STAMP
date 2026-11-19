'''
STAMP: 2D mocked wedge-aware alignment and weighted averaging
'''

# Import external dependencies
import numpy as np
import scipy.fft as sfft
from dataclasses import dataclass
from scipy.ndimage import rotate

# Import internal STAMP objects
from stamp.tools.diagram.utils.search import SearchSpace

# Alignment: best pose of every particle against every reference
@dataclass
class Alignment:
    cc: np.ndarray      # (n, K) normalised correlation
    angle: np.ndarray   # (n, K) degrees
    shift: np.ndarray   # (n, K, 2) (dy, dx) in working pixels

# bin_stack: block-mean a stack of images by an integer factor
def bin_stack(stack: np.ndarray, factor: int) -> np.ndarray:
    n, ny, nx = stack.shape
    return stack.reshape(n, ny // factor, factor, nx // factor, factor).mean(axis=(2, 4))

# soft_disc: circular window that stops edge effects in correlation
def soft_disc(size: int, radius_fraction: float = 0.45, edge_px: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2)
    return np.clip((radius_fraction * size - radius) / edge_px + 0.5, 0.0, 1.0).astype(np.float32)

# rotated_bank: every reference rotated through search angles, shape (K * n_angles, ny, nx)
def rotated_bank(references: np.ndarray, angles: np.ndarray) -> np.ndarray:
    return np.array([rotate(reference, angle, reshape=False, order=1, mode='constant') if angle else reference for reference in references for angle in angles])

# align_to_references: normalised cross-correlation of every particle against every rotated reference over all allowed shifts, using each particle's own wedge mask on both sides
def align_to_references(particles: np.ndarray, masks: np.ndarray, references: np.ndarray, space: SearchSpace, *, chunk: int = 16) -> Alignment:
    n, size = len(particles), particles.shape[-1]
    n_ref, n_angle = len(references), len(space.angles)
    window = np.arange(-space.max_shift, space.max_shift + 1)
    rows = np.array([0]) if space.axial_only else window
    window_y = rows % size
    window_x = window % size
    soft = soft_disc(size)
    bank = sfft.fft2(rotated_bank(references, space.angles) * soft).astype(np.complex64)
    cc = np.zeros((n, n_ref), dtype=np.float32)
    angle = np.zeros((n, n_ref), dtype=np.float32)
    shift = np.zeros((n, n_ref, 2), dtype=np.int32)
    for start in range(0, n, chunk):
        batch = slice(start, min(start + chunk, n))
        mask = masks[batch][:, None].copy()
        mask[..., 0, 0] = 0.0   # mean carries no structure and would dominate correlation
        particle_f = (sfft.fft2(particles[batch] * soft)[:, None] * mask).astype(np.complex64)
        bank_f = bank[None] * mask
        norm = np.sqrt((np.abs(particle_f) ** 2).sum(axis=(-2, -1)) * (np.abs(bank_f) ** 2).sum(axis=(-2, -1))) + 1e-12
        surface = sfft.ifft2(particle_f * np.conj(bank_f)).real * size * size / norm[..., None, None]
        surface = surface[..., window_y[:, None], window_x[None, :]].reshape(surface.shape[0], n_ref, n_angle, len(window_y), len(window_x))
        flat = surface.reshape(surface.shape[0], n_ref, -1)
        best = flat.argmax(axis=-1)
        a_index, y_index, x_index = np.unravel_index(best, (n_angle, len(window_y), len(window_x)))
        cc[batch] = np.take_along_axis(flat, best[..., None], axis=-1)[..., 0]
        angle[batch] = space.angles[a_index]
        shift[batch, :, 0] = rows[y_index]
        shift[batch, :, 1] = window[x_index]
    return Alignment(cc, angle, shift)

# apply_pose: bring a particle into its reference's frame, undoing shift then rotation
def apply_pose(image: np.ndarray, angle: float, shift: tuple[float, float]) -> np.ndarray:
    moved = np.roll(image, (-int(round(shift[0])), -int(round(shift[1]))), axis=(0, 1))
    return rotate(moved, -angle, reshape=False, order=1, mode='constant') if angle else moved

# weighted_average: average aligned particles frequency by frequency, each divided by number of particles that observed it
def weighted_average(aligned: np.ndarray, aligned_masks: np.ndarray) -> np.ndarray:
    total = np.fft.fft2(aligned).sum(axis=0)
    coverage = aligned_masks.sum(axis=0)
    return np.fft.ifft2(total / np.maximum(coverage, 0.25)).real

# align_stack: bring every particle into frame of one reference, returning aligned images and rotated wedge masks
def align_stack(particles: np.ndarray, masks: np.ndarray, angles: np.ndarray, shifts: np.ndarray, factor: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if len(particles) == 0:
        return particles.copy(), masks.copy()
    aligned = np.array([apply_pose(p, a, (dy * factor, dx * factor)) for p, a, (dy, dx) in zip(particles, angles, shifts)])
    shifted_masks = np.array([rotate(np.fft.fftshift(m), -a, reshape=False, order=0, mode='constant') for m, a in zip(masks, angles)])
    return aligned, np.fft.ifftshift(shifted_masks, axes=(-2, -1))
