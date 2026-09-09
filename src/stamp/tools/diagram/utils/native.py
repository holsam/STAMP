'''
STAMP: 2D mock of the stamp-native picker
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import map_coordinates

# Import internal STAMP objects
from stamp.tools.diagram.utils.config import NativePickerConfig
from stamp.tools.diagram.utils.geometry import non_max_suppression

# robust_normalise: median / MAD z-score, as stamp.picking.geometry.robust_normalise
def robust_normalise(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0.0:
        return (values - median) / (1.4826 * mad)
    std = values.std()
    return np.zeros_like(values) if std == 0.0 else (values - median) / std

# blank_membrane: set membrane voxels to the background median so the offset shell only scores density off the membrane surface
def blank_membrane(image: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32).copy()
    background = image[segmentation == 0]
    fill = float(np.median(background)) if background.size else 0.0
    out[segmentation == 1] = fill
    return out

# extract_surface_2d: points around each vesicle at ~spacing_px arc spacing with the outward unit normal; 2D analogue of marching_cubes + downsample
def extract_surface_2d(vesicles, spacing_px: float):
    points, normals, owner = [], [], []
    for index, v in enumerate(vesicles):
        step = max(spacing_px / v['r'], 1e-3)
        for theta in np.arange(0.0, 2 * np.pi, step):
            direction = np.array([np.cos(theta), np.sin(theta)])
            points.append([v['cx'] + v['r'] * direction[0], v['cy'] + v['r'] * direction[1]])
            normals.append(direction)
            owner.append(index)
    return np.array(points), np.array(normals), np.array(owner)

# exclude_near_boundary_2d: mask of points at least margin_px from every edge of a shape
def exclude_near_boundary_2d(points, shape, margin_px):
    lower = np.all(points >= margin_px, axis=1)
    upper = np.all(points <= (np.array(shape)[::-1] - 1 - margin_px), axis=1)
    return lower & upper

# sample_along_normals_2d: mean bilinear-interpolated density in a shell stepped out from each point along direction * normal
def sample_along_normals_2d(image, points, normals, off_min, off_max, n_samples, direction):
    offsets = np.linspace(off_min, off_max, n_samples)
    sample_xy = (points[None, :, :] + direction * offsets[:, None, None] * normals[None, :, :])
    rows = sample_xy[..., 1].ravel()
    cols = sample_xy[..., 0].ravel()
    values = map_coordinates(image, np.vstack([rows, cols]), order=1, mode='nearest')
    return values.reshape(n_samples, points.shape[0]).mean(axis=0)

# run_native_picker: the whole native-picker pipeline, returning every intermediate so a diagram can show each step
def run_native_picker(image, segmentation, vesicles, config: NativePickerConfig):
    blanked = blank_membrane(image, segmentation)
    spacing = config.to_voxels(config.surface_spacing_angstrom)
    vertices, normals, owner = extract_surface_2d(vesicles, spacing)

    margin = config.to_voxels(config.offset_max_angstrom)
    inside = exclude_near_boundary_2d(vertices, image.shape, margin)
    vertices, normals, owner = vertices[inside], normals[inside], owner[inside]

    off_min = config.to_voxels(config.offset_min_angstrom)
    off_max = config.to_voxels(config.offset_max_angstrom)

    faces = []
    for direction in (1, -1):
        densities = sample_along_normals_2d(blanked, vertices, normals, off_min, off_max, config.n_samples, direction)
        scores = robust_normalise(config.density_sign * densities)
        faces.append({
            'direction': direction,
            'points': vertices,
            'normals': direction * normals,
            'densities': densities,
            'scores': scores,
        })

    all_points = np.concatenate([f['points'] for f in faces])
    all_normals = np.concatenate([f['normals'] for f in faces])
    all_scores = np.concatenate([f['scores'] for f in faces])

    above = all_scores >= config.n_mad
    kept_local = non_max_suppression(
        all_points[above], all_scores[above],
        config.to_voxels(config.min_particle_distance_angstrom),
    ) if above.any() else np.array([], dtype=int)
    kept_index = np.where(above)[0][kept_local] if len(kept_local) else kept_local

    return {
        'blanked': blanked,
        'off_min_px': off_min,
        'off_max_px': off_max,
        'vertices': vertices,
        'normals': normals,
        'faces': faces,
        'all_points': all_points,
        'all_normals': all_normals,
        'all_scores': all_scores,
        'above_mask': above,
        'picks': all_points[kept_index] if len(kept_index) else np.empty((0, 2)),
        'pick_normals': all_normals[kept_index] if len(kept_index) else np.empty((0, 2)),
        'pick_scores': all_scores[kept_index] if len(kept_index) else np.empty(0),
        'config': config,
    }
