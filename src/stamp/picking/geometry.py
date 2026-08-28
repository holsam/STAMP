'''
STAMP: geometric utilities
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage import measure

# Reference axis that a particle's assigned orientation rotates onto the surface normal. +z in (x, y, z) output convention
REFERENCE_AXIS_XYZ = np.array([0.0, 0.0, 1.0])

# extract_surface: extract a membrane surface from a binary segmentation, returning (vertices, normals)
def extract_surface(segmentation: np.ndarray, level: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(segmentation > level):
        raise ValueError('Segmentation contains no voxels above the surface level; nothing to extract')
    try:
        vertices, faces, normals, _values = measure.marching_cubes(segmentation.astype(np.float32), level=level)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f'Surface extraction failed: {exc}') from exc
    if vertices.shape[0] == 0:
        raise ValueError('Surface extraction produced no vertices')
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return vertices, normals / lengths

# downsample_points: voxel-grid downsample keeping one point per cell of side spacing_voxels (so sampling density is set by the requested spacing rather than mesh resolution)
def downsample_points(points: np.ndarray, normals: np.ndarray, spacing_voxels: float) -> tuple[np.ndarray, np.ndarray]:
    if spacing_voxels <= 0:
        raise ValueError('spacing_voxels must be positive.')
    if points.shape[0] == 0:
        return points, normals
    cells = np.floor(points / spacing_voxels).astype(np.int64)
    # np.unique with return_index gives the first occurrence per cell if sorted stably; lexsort by original index preserves that
    _unique_cells, first_indices = np.unique(cells, axis=0, return_index=True)
    keep = np.sort(first_indices)
    return points[keep], normals[keep]

# sample_along_normals: mean tomogram density in a shell offset along each point's normal
def sample_along_normals(
    tomogram: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    offset_min_voxels: float,
    offset_max_voxels: float,
    n_samples: int,
    direction: int,
) -> np.ndarray:
    if direction not in (1, -1):
        raise ValueError('direction must be +1 or -1.')
    if n_samples < 1:
        raise ValueError('n_samples must be at least 1.')
    if offset_max_voxels < offset_min_voxels:
        raise ValueError('offset_max_voxels must be >= offset_min_voxels.')
    offsets = np.linspace(offset_min_voxels, offset_max_voxels, n_samples)
    # (n_samples, N, 3): each surface point stepped out to each offset
    sample_points = points[None, :, :] + direction * offsets[:, None, None] * normals[None, :, :]
    flat = sample_points.reshape(-1, 3).T  # (3, n_samples*N) in (z, y, x)
    densities = map_coordinates(
        tomogram.astype(np.float32), flat, order=1, mode='nearest'
    )
    return densities.reshape(n_samples, points.shape[0]).mean(axis=0)


# robust_normalise: median/MAD normalisation so score thresholds are comparable across tomograms
def robust_normalise(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0.0:
        return np.zeros_like(values)
    # 1.4826 makes MAD a consistent estimator of sigma for normal data.
    return (values - median) / (1.4826 * mad)


# non_maximum_suppression: greedy non-maximum suppression, highest score first, returning indices of kept points
def non_maximum_suppression(
    points: np.ndarray, scores: np.ndarray, min_distance_voxels: float
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.array([], dtype=np.int64)
    if min_distance_voxels <= 0:
        return np.arange(points.shape[0])
    tree = cKDTree(points)
    order = np.argsort(-scores)
    suppressed = np.zeros(points.shape[0], dtype=bool)
    kept: list[int] = []
    for index in order:
        if suppressed[index]:
            continue
        kept.append(int(index))
        for neighbour in tree.query_ball_point(points[index], r=min_distance_voxels):
            suppressed[neighbour] = True
    return np.array(sorted(kept), dtype=np.int64)

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


# exclude_near_boundary: boolean mask of points at least margin_voxels from every face of the volume
def exclude_near_boundary(points: np.ndarray, shape: tuple[int, ...], margin_voxels: float) -> np.ndarray:
    lower = np.all(points >= margin_voxels, axis=1)
    upper = np.all(points <= (np.array(shape) - 1 - margin_voxels), axis=1)
    return lower & upper
