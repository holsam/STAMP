'''
STAMP: 2D mocked alignment search space and missing wedge
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior

# SearchSpace: what alignment may vary for a prior (angles in degrees, shifts in working pixels)
@dataclass(frozen=True)
class SearchSpace:
    prior: OrientationPrior
    angles: np.ndarray
    max_shift: int
    axial_only: bool = False    # shifts only along axis (x)

# search_space: normal opens azimuth and a small shift, axis only polarity flip and axial shift, none the full rotation on a finer grid
def search_space(prior: OrientationPrior, *, angular_step: float = 10.0, max_shift: int = 3, axial_range: int = 6) -> SearchSpace:
    if prior is OrientationPrior.AXIS:
        return SearchSpace(prior, np.array([0.0, 180.0]), axial_range, axial_only=True)
    step = angular_step / 2 if prior is OrientationPrior.NONE else angular_step
    return SearchSpace(prior, np.arange(0.0, 360.0, step), max_shift)

# wedge_keep: Fourier mask (unshifted layout) that keeps observed directions; missing wedge lies within 90 - tilt_max degrees of the ky axis rotated by rotation_degrees
def wedge_keep(size: int, tilt_max_degrees: float, rotation_degrees: float = 0.0) -> np.ndarray:
    ky, kx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing='ij')
    theta = np.arctan2(ky, kx) - np.deg2rad(rotation_degrees)
    keep = np.abs(np.cos(theta)) >= np.sin(np.deg2rad(90.0 - tilt_max_degrees))
    keep[0, 0] = True
    return keep.astype(np.float32)

# apply_wedge: remove unobserved directions from an image
def apply_wedge(image: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.fft2(image) * keep).real
