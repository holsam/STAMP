'''
STAMP: geometric utilities
'''

# Import external dependencies
import numpy as np
from collections.abc import Sequence
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage import measure
from typing import Literal

# Import internal STAMP objects
from stamp.utils.log import log

# Reference axis that a particle's assigned orientation rotates onto the surface normal. +z in (x, y, z) output convention
REFERENCE_AXIS_XYZ = np.array([0.0, 0.0, 1.0])

# extract_surface: extract a membrane surface from a binary segmentation, returning (vertices, normals)
def extract_surface(segmentation: np.ndarray, level: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(segmentation > level):
        log.error('Segmentation contains no voxels above the surface level; nothing to extract')
        raise ValueError('Segmentation contains no voxels above the surface level; nothing to extract')
    try:
        vertices, faces, normals, _values = measure.marching_cubes(segmentation.astype(np.float32), level=level)
    except (RuntimeError, ValueError) as exc:
        log.error(f'Surface extraction failed: {exc}')
        raise ValueError(f'Surface extraction failed: {exc}') from exc
    if vertices.shape[0] == 0:
        log.error('Surface extraction produced no vertices')
        raise ValueError('Surface extraction produced no vertices')
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    log.debug(f'Extracted surface: {len(vertices)} vertices, {len(faces)} faces')
    return vertices, normals / lengths

# downsample_points: voxel-grid downsample keeping one point per cell of side spacing_voxels (so sampling density is set by the requested spacing rather than mesh resolution)
def downsample_points(points: np.ndarray, normals: np.ndarray, spacing_voxels: float) -> tuple[np.ndarray, np.ndarray]:
    if spacing_voxels <= 0:
        raise ValueError('spacing_voxels must be positive.')
    if points.shape[0] == 0:
        return points, normals
    cells = np.floor(points / spacing_voxels).astype(np.int64)
    # np.unique with return_index gives the lowest original index per unique cell
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
    *,
    return_samples: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
    reshaped = densities.reshape(n_samples, points.shape[0])  # (n_samples, N)
    mean = reshaped.mean(axis=0)
    if return_samples:
        return mean, reshaped.T  # profile: (N, n_samples), sample order matches offsets
    return mean

# profile_correlation_scores: Pearson correlation of each point's sign-corrected radial profile against a Gaussian bump centred at peak_offset_voxels with the given width
def profile_correlation_scores(
    profiles: np.ndarray,
    offsets_voxels: np.ndarray,
    peak_offset_voxels: float,
    width_voxels: float,
    density_sign: int,
) -> np.ndarray:
    if width_voxels <= 0:
        raise ValueError('width_voxels must be positive.')
    if profiles.shape[1] < 3:
        raise ValueError('Profile scoring needs at least 3 samples per point to be meaningful.')
    expected = np.exp(-0.5 * ((offsets_voxels - peak_offset_voxels) / width_voxels) ** 2)
    expected_centred = expected - expected.mean()
    expected_norm = np.sqrt(np.sum(expected_centred**2))
    if expected_norm == 0.0:
        raise ValueError('Expected profile has zero variance; widen width_voxels or move the peak inside the offset window.')
    signed = density_sign * profiles
    signed_centred = signed - signed.mean(axis=1, keepdims=True)
    numerator = signed_centred @ expected_centred
    signed_norm = np.sqrt(np.sum(signed_centred**2, axis=1))
    # points with a flat profile (zero variance) correlate with nothing; score them 0 rather than NaN
    return np.divide(
        numerator, signed_norm * expected_norm,
        out=np.zeros_like(numerator), where=signed_norm > 0,
    )

# robust_normalise: median/MAD normalisation so score thresholds are comparable across tomograms
def robust_normalise(
    values: np.ndarray,
    *,
    mode: Literal['global', 'local', 'vesicle'] = 'global',
    points: np.ndarray | None = None,
    local_radius_voxels: float | None = None,
    min_local_neighbours: int = 20,
    vesicle_ids: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == 'local':
        if points is None or local_radius_voxels is None:
            raise ValueError('local mode requires points and local_radius_voxels')
        return local_normalise(values, points, local_radius_voxels, min_local_neighbours)
    if mode == 'vesicle':
        if vesicle_ids is None:
            raise ValueError('vesicle mode requires vesicle_ids')
        return vesicle_normalise(values, vesicle_ids)

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0.0:
        # 1.4826 makes MAD a consistent estimator of sigma for normal data
        scaled = (values - median) / (1.4826 * mad)
    else:
        std = values.std()
        scaled = np.zeros_like(values) if std == 0.0 else (values - median) / std
    return scaled, np.zeros_like(values, dtype=bool)

# local_normalise: per-point median/MAD normalisation over a k-d tree neighbourhood
def local_normalise(
    values: np.ndarray,
    points: np.ndarray,
    radius_voxels: float,
    min_neighbours: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_points = points.shape[0]
    normalised = np.empty(n_points, dtype=np.float64)
    used_fallback = np.zeros(n_points, dtype=bool)

    global_median = np.median(values)
    global_mad = np.median(np.abs(values - global_median))
    global_scale = 1.4826 * global_mad if global_mad > 0.0 else (values.std() or 1.0)

    tree = cKDTree(points)
    neighbour_lists = tree.query_ball_point(points, r=radius_voxels)

    for index, neighbours in enumerate(neighbour_lists):
        if len(neighbours) < min_neighbours:
            used_fallback[index] = True
            normalised[index] = (values[index] - global_median) / global_scale
            continue
        local_values = values[neighbours]
        local_median = np.median(local_values)
        local_mad = np.median(np.abs(local_values - local_median))
        local_scale = 1.4826 * local_mad if local_mad > 0.0 else global_scale
        normalised[index] = (values[index] - local_median) / local_scale

    return normalised, used_fallback

# vesicle_normalise: per-vesicle median/MAD normalisation; points with no vesicle_id ('') fall back to the pooled/global estimate
def vesicle_normalise(values: np.ndarray, vesicle_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(vesicle_ids)
    normalised, _fallback = robust_normalise(values)
    used_fallback = np.zeros(values.shape[0], dtype=bool)
    for group in set(groups):
        if not group:
            continue
        mask = groups == group
        normalised[mask], _ = robust_normalise(values[mask])
    used_fallback[groups == ''] = True
    return normalised, used_fallback

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

# exclude_near_boundary: boolean mask of points at least margin_voxels from every face of the volume
def exclude_near_boundary(points: np.ndarray, shape: tuple[int, ...], margin_voxels: float) -> np.ndarray:
    lower = np.all(points >= margin_voxels, axis=1)
    upper = np.all(points <= (np.array(shape) - 1 - margin_voxels), axis=1)
    return lower & upper

# score_membrane_faces: sample both faces of every surface point across multiple offset windows, scoring each window by mean density and by radial-profile shape
def score_membrane_faces(
    tomogram: np.ndarray,
    vertices: np.ndarray,
    normals: np.ndarray,
    offset_windows_voxels: Sequence[tuple[float, float]],
    n_samples: int,
    density_sign: int,
    *,
    scoring_mode: Literal['mean', 'profile'] = 'mean',
    profile_width_voxels: float | None = None,
    mode: Literal['global', 'local', 'vesicle'] = 'global',
    local_radius_voxels: float | None = None,
    min_local_neighbours: int = 20,
    vesicle_ids: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(offset_windows_voxels) == 0:
        raise ValueError('offset_windows_voxels must contain at least one window.')

    points = np.concatenate([vertices, vertices])
    face_normals = np.concatenate([normals, -normals])

    n_windows = len(offset_windows_voxels)
    window_scores = np.empty((n_windows, points.shape[0]))
    window_mean_scores = np.empty((n_windows, points.shape[0]))
    window_profile_scores = np.empty((n_windows, points.shape[0]))
    window_fallback = np.empty((n_windows, points.shape[0]), dtype=bool)
    for window_index, (offset_min_voxels, offset_max_voxels) in enumerate(offset_windows_voxels):
        means, profiles = zip(*[sample_along_normals(tomogram, vertices, normals, offset_min_voxels, offset_max_voxels, n_samples, direction, return_samples=True) for direction in (1, -1)])
        raw_means = density_sign * np.concatenate(means)
        pooled_profiles = np.concatenate(profiles, axis=0)

        offsets_voxels = np.linspace(offset_min_voxels, offset_max_voxels, n_samples)
        peak_voxels = (offset_min_voxels + offset_max_voxels) / 2.0  # window's own midpoint
        width_voxels = profile_width_voxels if profile_width_voxels is not None else (offset_max_voxels - offset_min_voxels) / 4.0
        raw_profile = profile_correlation_scores(pooled_profiles, offsets_voxels, peak_voxels, width_voxels, density_sign)

        window_mean_scores[window_index], mean_fallback = robust_normalise(raw_means, mode=mode, points=points, local_radius_voxels=local_radius_voxels, min_local_neighbours=min_local_neighbours, vesicle_ids=vesicle_ids)
        window_profile_scores[window_index], profile_fallback = robust_normalise(raw_profile, mode=mode, points=points, local_radius_voxels=local_radius_voxels, min_local_neighbours=min_local_neighbours, vesicle_ids=vesicle_ids)
        if scoring_mode == 'profile':
            window_scores[window_index], window_fallback[window_index] = window_profile_scores[window_index], profile_fallback
        else:
            window_scores[window_index], window_fallback[window_index] = window_mean_scores[window_index], mean_fallback

    winning_window = np.argmax(window_scores, axis=0)
    point_index = np.arange(points.shape[0])
    scores = window_scores[winning_window, point_index]
    mean_scores = window_mean_scores[winning_window, point_index]
    profile_scores = window_profile_scores[winning_window, point_index]
    used_fallback = window_fallback[winning_window, point_index]
    return points, face_normals, scores, winning_window, used_fallback, mean_scores, profile_scores

# max_order_statistic_offset: approximate upward bias, in normalised score units, of the maximum of n_windows independent standard-normal scores
def max_order_statistic_offset(n_windows: int) -> float:
    if n_windows < 1:
        raise ValueError('n_windows must be at least 1.')
    if n_windows == 1:
        return 0.0
    return float(np.sqrt(2.0 * np.log(n_windows)))

# beam_angle_deviation_degrees: angle between an outward normal and the beam-orthogonal plane (0 = beam-orthogonal, 90 = beam-aligned)
def beam_angle_deviation_degrees(normal_xyz: np.ndarray) -> float:
    norm = np.linalg.norm(normal_xyz)
    if norm == 0.0:
        raise ValueError('normal_xyz must be non-zero.')
    z_component = abs(normal_xyz[2] / norm)
    return float(np.degrees(np.arcsin(np.clip(z_component, 0.0, 1.0))))
