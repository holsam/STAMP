'''
STAMP: native membrane-guided, template-free picker
'''

# Import external dependencies
import mrcfile, numpy as np
from dataclasses import dataclass

# Import internal STAMP objects
from stamp.picking.geometry import (
    downsample_points,
    exclude_near_boundary,
    extract_surface,
    non_maximum_suppression,
    quaternion_from_reference_to,
    robust_normalise,
    sample_along_normals,
)
from stamp.schemas.picks import RawPick

# PICKER_NAME: name for picker tool
PICKER_NAME = 'stamp-native'

# NativePickerConfig: parameters for one stamp-native run (all distances in Angstrom)
@dataclass(frozen=True)
class NativePickerConfig:
    voxel_size_angstrom: float
    # radial window from membrane surface to search
    offset_min_angstrom: float = 20.0
    offset_max_angstrom: float = 80.0
    # sample per surface point within radial window
    n_samples: int = 5
    # sampling point spacing
    surface_spacing_angstrom: float = 20.0
    # minimum particle separation
    min_particle_distance_angstrom: float = 60.0
    # score threshold
    n_mad: float = 3.0
    # protein/background density
    density_sign: int = -1
    # boundary for dropping surface points
    boundary_margin_angstrom: float | None = None

    def __post_init__(self) -> None:
        if self.voxel_size_angstrom <= 0:
            raise ValueError('voxel_size_angstrom must be positive.')
        if self.density_sign not in (1, -1):
            raise ValueError('density_sign must be +1 or -1.')
        if self.offset_max_angstrom < self.offset_min_angstrom:
            raise ValueError('offset_max_angstrom must be >= offset_min_angstrom.')

    def to_voxels(self, angstrom: float) -> float:
        return angstrom / self.voxel_size_angstrom

# pick_tomogram: run the native picker over one tomogram, returning RawPicks
def pick_tomogram(
    segmentation_path,
    tomogram_path,
    tomogram_id: str,
    config: NativePickerConfig,
) -> list[RawPick]:
    with mrcfile.open(str(segmentation_path), permissive=True) as mrc:
        segmentation = np.asarray(mrc.data)
    with mrcfile.open(str(tomogram_path), permissive=True) as mrc:
        tomogram = np.asarray(mrc.data)

    if segmentation.shape != tomogram.shape:
        raise ValueError(f'Segmentation shape {segmentation.shape} does not match tomogram shape {tomogram.shape} for {tomogram_id}; they must be the same volume at the same binning')

    # Blank membrane signal so offset shell only scores densities off membrane surface
    tomogram = tomogram.astype(np.float32).copy()
    tomogram[segmentation > 0] = np.median(tomogram[segmentation <= 0])

    vertices, normals = extract_surface(segmentation)
    vertices, normals = downsample_points(vertices, normals, config.to_voxels(config.surface_spacing_angstrom))

    margin_angstrom = config.boundary_margin_angstrom if config.boundary_margin_angstrom is not None else config.offset_max_angstrom
    inside = exclude_near_boundary(vertices, segmentation.shape, config.to_voxels(margin_angstrom))
    vertices, normals = vertices[inside], normals[inside]
    if vertices.shape[0] == 0:
        return []

    offset_min = config.to_voxels(config.offset_min_angstrom)
    offset_max = config.to_voxels(config.offset_max_angstrom)

    # score both membrane faces independently
    candidate_points: list[np.ndarray] = []
    candidate_normals: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []

    for direction in (1, -1):
        densities = sample_along_normals(
            tomogram, vertices, normals, offset_min, offset_max,
            config.n_samples, direction,
        )
        scores = robust_normalise(config.density_sign * densities)
        candidate_points.append(vertices)
        candidate_normals.append(direction * normals)
        candidate_scores.append(scores)

    points = np.concatenate(candidate_points)
    outward_normals = np.concatenate(candidate_normals)
    scores = np.concatenate(candidate_scores)

    above_threshold = scores >= config.n_mad
    points, outward_normals, scores = (
        points[above_threshold],
        outward_normals[above_threshold],
        scores[above_threshold],
    )
    if points.shape[0] == 0:
        return []

    kept = non_maximum_suppression(points, scores, config.to_voxels(config.min_particle_distance_angstrom))

    picks: list[RawPick] = []
    for index in kept:
        # convert internal (z, y, x) -> output (x, y, z)
        position_zyx = points[index]
        normal_zyx = outward_normals[index]
        position_xyz = (
            float(position_zyx[2]), float(position_zyx[1]), float(position_zyx[0])
        )
        normal_xyz = np.array([normal_zyx[2], normal_zyx[1], normal_zyx[0]])

        picks.append(
            RawPick(
                tomogram_id=tomogram_id,
                position=position_xyz,
                orientation=quaternion_from_reference_to(normal_xyz),
                confidence=float(scores[index]),
                source_picker=PICKER_NAME,
            )
        )
    return picks
