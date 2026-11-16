'''
STAMP: classify settings shared by all orientation priors
'''

# Import external dependencies
from dataclasses import dataclass

# Import STAMP objects
from stamp.classify.cluster import ClusteringConfig

# GroupSettings: classify settings shared by all orientation priors
@dataclass(frozen=True)
class GroupSettings:
    box_voxels: int
    voxel_size_angstrom: float
    n_radial_bins: int
    azimuthal_modes: int
    n_azimuthal_samples: int
    min_radius_fraction: float
    feature_normalisation: str
    clustering: ClusteringConfig
    strict_halfset_independence: bool
    inplane_angular_step_degrees: float
    n_rotations: int
    shift_range_voxels: int
    polish_rounds: int
    min_cluster_size: int
    merge_fraction: float
    null_sigmas: float
    reject_quantile: float
    random_state: int
    bin_factor: int
    tilt_max_degrees: float | None
    tilt_axis_degrees: float
    n_workers: int
