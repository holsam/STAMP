'''
STAMP: synthetic particles with known orientations, shifts and wedges
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import gaussian_filter, shift as shift_volume

# Import internal STAMP objects
from stamp.classify.frames import rotate_density
from stamp.classify.wedge import apply_wedge

# asymmetric_object: smooth volume of random gaussian blobs, no symmetry, so every rotation and flip is distinguishable
def asymmetric_object(box: int, seed: int, *, n_blobs: int = 7, spread: float = 0.22) -> np.ndarray:
    rng = np.random.default_rng(seed)
    volume = np.zeros((box,) * 3, dtype=np.float32)
    centre = (box - 1) / 2.0
    for _ in range(n_blobs):
        index = np.round(centre + rng.normal(0.0, spread * box, 3)).astype(int).clip(2, box - 3)
        volume[tuple(index)] += rng.uniform(0.5, 1.5)
    smooth = gaussian_filter(volume, box / 14.0)
    return smooth / smooth.max()

# rod_object: filament along z with a repeating asymmetric pattern, polarity and axial position are both observable
def rod_object(box: int, seed: int, *, repeat: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centre = (box - 1) / 2.0
    axis = np.arange(box) - centre
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    body = np.exp(-(x**2 + y**2) / (2 * (box / 10.0) ** 2))
    phase = rng.uniform(0, 2 * np.pi)
    lump = np.exp(-((x - box / 7.0 * np.cos(2 * np.pi * z / repeat + phase)) ** 2 + (y - box / 7.0 * np.sin(2 * np.pi * z / repeat + phase)) ** 2) / (2 * (box / 16.0) ** 2))
    cap = 2.0 * np.exp(-((x - box / 5.0) ** 2 + y**2 + (z - box / 4.0) ** 2) / (2 * (box / 12.0) ** 2))     # off-axis cap at one end breaks the up-down symmetry so polarity is observable
    return (body + lump + cap).astype(np.float32)

# make_particle: the object moved by rotation then shift, limited to the box-frame wedge, plus white noise
def make_particle(
    obj: np.ndarray,
    rotation: np.ndarray,
    shift: tuple[float, float, float],
    wedge: np.ndarray | None,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    moved = shift_volume(rotate_density(obj, rotation), shift, order=1)
    if wedge is not None:
        moved = apply_wedge(moved, wedge)
    return (moved + rng.normal(0.0, noise_sigma, obj.shape)).astype(np.float32)
