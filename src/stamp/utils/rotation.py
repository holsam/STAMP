'''
STAMP: rotation conversions between unit quaternions (w, x, y, z) and rotation matrices
'''

# Import external dependencies
import numpy as np

# REFERENCE_AXIS_XYZ: reference axis that a particle's assigned orientation rotates onto the surface normal. +z in (x, y, z) output convention
REFERENCE_AXIS_XYZ = np.array([0.0, 0.0, 1.0])

# quaternion_to_matrix: return a rotation matrix from a unit quaternion (w, x, y, z)
def quaternion_to_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )

# quaternion_from_reference_to: unit quaternion rotating REFERENCE_AXIS_XYZ (+z) onto the given vector
def quaternion_from_reference_to(vector_xyz: np.ndarray) -> tuple[float, float, float, float]:
    norm = np.linalg.norm(vector_xyz)
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    target = vector_xyz / norm
    dot = float(np.dot(REFERENCE_AXIS_XYZ, target))
    if dot < -1.0 + 1e-8:
        # antiparallel: 180 degrees about any perpendicular axis
        return (0.0, 1.0, 0.0, 0.0)
    cross = np.cross(REFERENCE_AXIS_XYZ, target)
    quaternion = np.array([1.0 + dot, cross[0], cross[1], cross[2]])
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(component) for component in quaternion)

# normal_from_quaternion: the image of +z under a (w, x, y, z) unit quaternion
def normal_from_quaternion(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
