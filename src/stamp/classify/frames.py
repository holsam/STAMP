'''
STAMP: frame and rotation conventions shared by classification
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

# azimuth_matrix: right-handed rotation by angle_degrees about the box z axis (x towards y), acting on (x, y, z) box coordinates
def azimuth_matrix(angle_degrees: float) -> np.ndarray:
    angle = np.radians(angle_degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])

# azimuth_degrees: signed rotation angle about z of a matrix that is a pure z rotation, in (-180, 180]
def azimuth_degrees(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))

# flip_matrix: 180 degree rotation about the box x axis, which sends +z to -z (polarity flip of axis prior)
def flip_matrix() -> np.ndarray:
    return np.diag([1.0, -1.0, -1.0])

# rotate_density: move the density of a box by a rotation about its centre, so a feature at offset r from the centre ends at rotation @ r (array axes are the box x, y, z)
def rotate_density(volume: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    centre = (np.array(volume.shape) - 1) / 2.0
    matrix = np.asarray(rotation).T     # affine_transform maps output to input coordinates
    return affine_transform(volume, matrix, offset=centre - matrix @ centre, order=1)

# rotation_to_quaternion: unit quaternion (w, x, y, z) of a rotation matrix, with w >= 0 so the sign is unique
def rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    sign = -1.0 if w < 0 else 1.0
    return (float(sign * w), float(sign * x), float(sign * y), float(sign * z))
