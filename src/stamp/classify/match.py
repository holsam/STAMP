'''
STAMP: wedge-aware alignment of particles to a reference over a search space
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass

# Import internal STAMP objects
from stamp.classify.search import SearchSpace, rotated_bank

# Alignment: best candidate per particle against one reference
@dataclass(frozen=True)
class Alignment:
    rotation_index: np.ndarray  # (n,), index into SearchSpace.rotations
    shift: np.ndarray           # (n, 3), integer working voxels, particle = shift(rotate(reference))
    cc: np.ndarray              # (n,), normalised correlation over the particle's observed Fourier components

# wedge_filter: particles limited to their own observed Fourier components, so correlation against an unmasked reference equals correlation against a masked one
def wedge_filter(particles: np.ndarray, wedges: np.ndarray) -> np.ndarray:
    return np.fft.ifftn(np.fft.fftn(particles, axes=(1, 2, 3)) * wedges, axes=(1, 2, 3)).real.astype(np.float32)

# EXHAUSTIVE_BELOW: search spaces up to this size get shift search for every rotation, because shift-free score cannot rank rotations when shift window is large (axis prior)
EXHAUSTIVE_BELOW = 128

# rank_rotations: normalised correlation and masked reference norm, both (n_particles, n_rotations), without shifts (masked norm of a rotated reference is its wedge-weighted Fourier power)
def rank_rotations(filtered: np.ndarray, wedges: np.ndarray, bank: np.ndarray, box_voxels: int, *, chunk: int = 250) -> tuple[np.ndarray, np.ndarray]:
    flat = filtered.reshape(len(filtered), -1)
    flat_wedges = wedges.reshape(len(wedges), -1)
    particle_norms = np.linalg.norm(flat, axis=1)[:, None] + 1e-8
    scores = np.empty((len(flat), len(bank)), dtype=np.float32)
    norms = np.empty_like(scores)
    for start in range(0, len(bank), chunk):
        part = bank[start:start + chunk]
        power = np.abs(np.fft.fftn(part.reshape(-1, *(box_voxels,) * 3), axes=(1, 2, 3))) ** 2
        masked_norms = np.sqrt(flat_wedges @ power.reshape(len(part), -1).T.astype(np.float32) / flat_wedges.shape[1]) + 1e-8
        scores[:, start:start + chunk] = (flat @ part.T) / (particle_norms * masked_norms)
        norms[:, start:start + chunk] = masked_norms
    return scores, norms

# _window: wrapped FFT indices of the shifts -m..m
def _window(max_shift: int, box_voxels: int) -> np.ndarray:
    return np.arange(-max_shift, max_shift + 1) % box_voxels

# refine_shift: best shift inside window for one particle against a few candidate rotations
def refine_shift(
    particle_ft: np.ndarray,
    filtered_norm: float,
    candidates_ft: np.ndarray,
    masked_norms: np.ndarray,
    max_shift: tuple[int, int, int],
) -> tuple[int, np.ndarray, float]:
    box_voxels = particle_ft.shape[0]
    correlation = np.fft.ifftn(particle_ft[None] * np.conj(candidates_ft), axes=(1, 2, 3)).real
    windows = [_window(m, box_voxels) for m in max_shift]
    local = correlation[:, windows[0][:, None, None], windows[1][None, :, None], windows[2][None, None, :]]
    flat = local.reshape(len(local), -1)
    peaks = flat.argmax(axis=1)
    scores = flat[np.arange(len(flat)), peaks] / (filtered_norm * masked_norms)
    best = int(scores.argmax())
    offsets = np.unravel_index(peaks[best], local.shape[1:])
    return best, np.array([o - m for o, m in zip(offsets, max_shift)]), float(scores[best])

# align_to_reference: best rotation and shift of every particle against one reference (stage one ranks rotations without shifts, stage two searches shifts for the `top` best)
def align_to_reference(
    particles: np.ndarray,
    wedges: np.ndarray,
    reference: np.ndarray,
    space: SearchSpace,
    *,
    top: int | None = None,
    threads: int = 1,
    bank: np.ndarray | None = None,
) -> Alignment:
    box_voxels = particles.shape[1]
    bank = rotated_bank(reference, space.rotations, threads=threads) if bank is None else bank
    filtered = wedge_filter(particles, wedges)
    scores, norms = rank_rotations(filtered, wedges, bank, box_voxels)
    top = (len(bank) if len(bank) <= EXHAUSTIVE_BELOW else 5) if top is None else min(top, len(bank))
    shortlist = np.tile(np.arange(top), (len(particles), 1)) if top == len(bank) else np.argsort(-scores, axis=1)[:, :top]   # exhaustive keeps bank order so one set of spectra serves every particle
    bank_ft = np.fft.fftn(bank.reshape(len(bank), *(box_voxels,) * 3), axes=(1, 2, 3)) if top == len(bank) else None
    rotation_index = np.empty(len(particles), dtype=np.int64)
    shift = np.zeros((len(particles), 3), dtype=np.int64)
    cc = np.empty(len(particles), dtype=np.float32)
    bank_cubes = bank.reshape(len(bank), *(box_voxels,) * 3)
    for i in range(len(particles)):
        candidates_ft = bank_ft if bank_ft is not None else np.fft.fftn(bank_cubes[shortlist[i]], axes=(1, 2, 3))
        best, shift[i], cc[i] = refine_shift(np.fft.fftn(filtered[i]), np.linalg.norm(filtered[i]) + 1e-8, candidates_ft, norms[i, shortlist[i]], space.max_shift)
        rotation_index[i] = shortlist[i, best]
    return Alignment(rotation_index, shift, cc)
