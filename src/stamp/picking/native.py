'''
STAMP: native membrane-guided, template-free picker
'''

# Import external dependencies
import mrcfile, numpy as np
from dataclasses import dataclass
from typing import Literal

# Import internal STAMP objects
from stamp.picking.geometry import (
    beam_angle_deviation_degrees,
    downsample_points,
    exclude_near_boundary,
    extract_surface,
    max_order_statistic_offset,
    non_maximum_suppression,
    quaternion_from_reference_to,
    robust_normalise,
    sample_along_normals,
    score_membrane_faces
)
from stamp.schemas.picks import RawPick
from stamp.utils.log import log

# PICKER_NAME: name for picker tool
PICKER_NAME = 'stamp-native'

# DEFAULT_OFFSET_WINDOWS_ANGSTROM: (min, max) radial windows
DEFAULT_OFFSET_WINDOWS_ANGSTROM: tuple[tuple[float, float], ...] = (
    (20.0, 50.0),
    (40.0, 80.0),
    (70.0, 110.0),
    (100.0, 160.0),
)

# NativePickerConfig: parameters for one stamp-native run (all distances in Angstrom)
@dataclass(frozen=True)
class NativePickerConfig:
    voxel_size_angstrom: float
    # radial windows from membrane surface to search
    offset_windows_angstrom: tuple[tuple[float, float], ...] = DEFAULT_OFFSET_WINDOWS_ANGSTROM
    # sample per surface point within each radial window
    n_samples: int = 5
    # sampling point spacing
    surface_spacing_angstrom: float = 20.0
    # minimum particle separation
    min_particle_distance_angstrom: float = 60.0
    # score threshold before multi-window order-statistic correction
    n_mad: float = 3.0
    # protein/background density
    density_sign: int = -1
    # boundary for dropping surface points; defaults to the largest window's outer edge
    boundary_margin_angstrom: float | None = None
    # score normalisation to use
    normalisation: Literal['global', 'local'] = 'global'
    # neighbourhood radius for local normalisation
    local_radius_angstrom: float = 150.0
    # minimum neighbours within local_radius_angstrom before falling back to global
    min_local_neighbours: int = 20
    # scoring mode to use for particle radial density
    scoring_mode: Literal['mean', 'profile'] = 'mean'
    # width (Angstrom) of the expected profile bump
    profile_width_angstrom: float | None = None

    def __post_init__(self) -> None:
        if self.voxel_size_angstrom <= 0:
            raise ValueError('voxel_size_angstrom must be positive.')
        if self.density_sign not in (1, -1):
            raise ValueError('density_sign must be +1 or -1.')
        if len(self.offset_windows_angstrom) == 0:
            raise ValueError('offset_windows_angstrom must contain at least one window.')
        for offset_min, offset_max in self.offset_windows_angstrom:
            if offset_max < offset_min:
                raise ValueError('each offset window must have offset_max >= offset_min.')
        if self.local_radius_angstrom <= 0:
            raise ValueError('local_radius_angstrom must be positive.')
        if self.min_local_neighbours < 1:
            raise ValueError('min_local_neighbours must be at least 1.')
        if self.scoring_mode not in ('mean', 'profile'):
            raise ValueError('scoring_mode must be "mean" or "profile".')
        if self.profile_width_angstrom is not None and self.profile_width_angstrom <= 0:
            raise ValueError('profile_width_angstrom must be positive.')

    def to_voxels(self, angstrom: float) -> float:
        return angstrom / self.voxel_size_angstrom

    # effective_n_mad: n_mad corrected for taking the best of len(offset_windows_angstrom) scores
    @property
    def effective_n_mad(self) -> float:
        return self.n_mad + max_order_statistic_offset(len(self.offset_windows_angstrom))

    # max_offset_angstrom: outer edge of the widest window, used as the default boundary margin
    @property
    def max_offset_angstrom(self) -> float:
        return max(offset_max for _offset_min, offset_max in self.offset_windows_angstrom)

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
        log.error(f'Segmentation shape {segmentation.shape} does not match tomogram shape {tomogram.shape} for {tomogram_id}')
        raise ValueError(f'Segmentation shape {segmentation.shape} does not match tomogram shape {tomogram.shape} for {tomogram_id}; they must be the same volume at the same binning')

    # Blank membrane signal so offset shell only scores densities off membrane surface
    tomogram = tomogram.astype(np.float32).copy()
    tomogram[segmentation > 0] = np.median(tomogram[segmentation <= 0])

    vertices, normals = extract_surface(segmentation)
    vertices, normals = downsample_points(vertices, normals, config.to_voxels(config.surface_spacing_angstrom))

    margin_angstrom = config.boundary_margin_angstrom if config.boundary_margin_angstrom is not None else config.max_offset_angstrom
    inside = exclude_near_boundary(vertices, segmentation.shape, config.to_voxels(margin_angstrom))
    vertices, normals = vertices[inside], normals[inside]
    if vertices.shape[0] == 0:
        return []

    offset_windows_voxels = [(config.to_voxels(offset_min), config.to_voxels(offset_max)) for offset_min, offset_max in config.offset_windows_angstrom]

    profile_width_voxels = (config.to_voxels(config.profile_width_angstrom) if config.profile_width_angstrom is not None else None)
    points, outward_normals, scores, winning_window, used_fallback, mean_scores, profile_scores = score_membrane_faces(
        tomogram, vertices, normals,
        offset_windows_voxels,
        config.n_samples, config.density_sign,
        scoring_mode=config.scoring_mode,
        profile_width_voxels=profile_width_voxels,
        mode=config.normalisation,
        local_radius_voxels=config.to_voxels(config.local_radius_angstrom),
        min_local_neighbours=config.min_local_neighbours,
    )
    above_threshold = scores >= config.effective_n_mad
    n_candidates = points.shape[0]
    points, outward_normals, scores, winning_window, used_fallback, mean_scores, profile_scores = (
        points[above_threshold],
        outward_normals[above_threshold],
        scores[above_threshold],
        winning_window[above_threshold],
        used_fallback[above_threshold],
        mean_scores[above_threshold],
        profile_scores[above_threshold],
    )
    log.debug(f'{tomogram_id}: {n_candidates} candidates before filtering, {points.shape[0]} kept')
    if points.shape[0] == 0:
        return []
    if config.normalisation == 'local' and used_fallback.any():
        log.debug(f'{tomogram_id}: {int(used_fallback.sum())}/{points.shape[0]} kept points used global fallback (too few neighbours within local_radius_angstrom)')

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
                metadata={'used_global_fallback': bool(used_fallback[index])} if config.normalisation == 'local' else {},
                beam_angle_deviation_degrees=beam_angle_deviation_degrees(normal_xyz),
                offset_window_angstrom=config.offset_windows_angstrom[int(winning_window[index])],
                mean_score=float(mean_scores[index]),
                profile_score=float(profile_scores[index]),
            )
        )
    return picks
