'''
STAMP: per-particle missing wedge in the box frame
'''

# Import external dependencies
import numpy as np

# Import internal STAMP objects
from stamp.classify.extract import quaternion_to_matrix

# _tilt_normal: in-plane tomogram direction perpendicular to the tilt axis
def _tilt_normal(tilt_axis_degrees: float) -> np.ndarray:
    axis = np.radians(tilt_axis_degrees)
    return np.array([np.sin(axis), -np.cos(axis), 0.0])

# observed_directions: whether Fourier vectors (n, 3) given in the box frame survive the wedge; margin_degrees shrinks the tilt limit to stay clear of the soft edge
def observed_directions(
    vectors_xyz: np.ndarray,
    orientation: tuple[float, float, float, float] | None,
    tilt_max_degrees: float | None,
    tilt_axis_degrees: float,
    *,
    margin_degrees: float = 0.0,
) -> np.ndarray:
    if tilt_max_degrees is None:
        return np.ones(len(vectors_xyz), dtype=bool)
    tomogram_frame = vectors_xyz if orientation is None else vectors_xyz @ quaternion_to_matrix(orientation).T
    limit = np.tan(np.radians(max(tilt_max_degrees - margin_degrees, 1.0)))
    return np.abs(tomogram_frame[:, 2]) <= np.abs(tomogram_frame @ _tilt_normal(tilt_axis_degrees)) * limit

# box_wedge: float32 0/1 mask over the FFT grid of a box (unshifted order, axes x, y, z), 1 where the particle's Fourier components were measured
def box_wedge(
    box_voxels: int,
    orientation: tuple[float, float, float, float] | None,
    tilt_max_degrees: float | None,
    tilt_axis_degrees: float,
) -> np.ndarray:
    if tilt_max_degrees is None:
        return np.ones((box_voxels,) * 3, dtype=np.float32)
    freq = np.fft.fftfreq(box_voxels)
    grid = np.stack(np.meshgrid(freq, freq, freq, indexing='ij'), axis=-1).reshape(-1, 3)
    observed = observed_directions(grid, orientation, tilt_max_degrees, tilt_axis_degrees)
    return observed.reshape((box_voxels,) * 3).astype(np.float32)

# apply_wedge: remove the unobserved Fourier components of a box
def apply_wedge(volume: np.ndarray, wedge: np.ndarray) -> np.ndarray:
    return np.fft.ifftn(np.fft.fftn(volume) * wedge).real.astype(np.float32)
