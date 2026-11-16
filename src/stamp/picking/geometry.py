'''
STAMP: geometric utilities
'''

# Import external dependencies
import numpy as np
from skimage import measure

# Import internal STAMP objects
from stamp.utils.errors import StampValidationError
from stamp.utils.log import log

# mesh_surface: marching cubes on a binary segmentation at level, returning (vertices, faces, normals); StampValidationError when there is no surface
def mesh_surface(segmentation: np.ndarray, level: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.any(segmentation > level):
        raise StampValidationError('Segmentation contains no voxels above the surface level; nothing to extract')
    try:
        vertices, faces, normals, _values = measure.marching_cubes(segmentation.astype(np.float32), level=level)
    except (RuntimeError, ValueError) as exc:
        raise StampValidationError(f'Surface extraction failed: {exc}') from exc
    if vertices.shape[0] == 0:
        raise StampValidationError('Surface extraction produced no vertices')
    log.debug(f'Extracted surface: {len(vertices)} vertices, {len(faces)} faces')
    return vertices, faces, normals

# extract_surface: extract a membrane surface from a binary segmentation, returning (vertices, normals)
def extract_surface(segmentation: np.ndarray, level: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    vertices, _faces, normals = mesh_surface(segmentation, level)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return vertices, normals / lengths

# compose_roll_about_normal: quaternion for "roll by angle_degrees about +z, then map +z onto the normal"
def compose_roll_about_normal(
    normal_quaternion: tuple[float, float, float, float],
    angle_degrees: float,
) -> tuple[float, float, float, float]:
    half = np.radians(angle_degrees) / 2.0
    roll = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # rotation about +z, (w, x, y, z)
    w0, x0, y0, z0 = normal_quaternion
    w1, x1, y1, z1 = roll
    product = np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])
    product /= np.linalg.norm(product) or 1.0
    return tuple(float(component) for component in product)

# beam_angle_deviation_degrees: angle between an outward normal and the beam-orthogonal plane (0 = beam-orthogonal, 90 = beam-aligned)
def beam_angle_deviation_degrees(normal_xyz: np.ndarray) -> float:
    norm = np.linalg.norm(normal_xyz)
    if norm == 0.0:
        raise ValueError('normal_xyz must be non-zero.')
    z_component = abs(normal_xyz[2] / norm)
    return float(np.degrees(np.arcsin(np.clip(z_component, 0.0, 1.0))))
