'''
STAMP: configuration dataclasses for the workflow diagrams
'''

# Import external dependencies
from dataclasses import dataclass, field
from pathlib import Path

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior

# DiagramConfig: class containing all config values
@dataclass
class DiagramConfig:
    # structures
    target: Path                                  # PDB recovered by identify (class 0)
    distractor: Path                              # known different species (class 1)
    also: list[Path] = field(default_factory=list)  # extra PDBs, NOT in the identify panel
    n_synthetic_unknowns: int = 2                 # procedurally generated extra classes
    # output
    outdir: Path = Path('out/diagrams')
    # geometry
    box_size: int = 64
    pixel_size: float = 3.0
    field_px: int = 900                           # side of the mock tomogram slice, pixels
    # orientation prior: sets the carriers particles sit on and the alignment search space
    orientation_prior: OrientationPrior = OrientationPrior.NORMAL
    tilt_max_degrees: float = 60.0                # half-width of the tilt range, sets the missing wedge
    # scene composition
    n_vesicles: int = 14
    n_target: int = 300
    n_distractor: int = 240
    n_extra: int = 100                            # particles per extra / unknown class
    n_filaments: int = 26                         # filaments in the axis scene
    # pick
    max_surface_distance_a: float = 72.0          # Å; picks further than this from a surface stay unoriented (normal prior)
    max_beam_angle_deviation: float = 60.0        # picks tilted further than this from beam-orthogonal are dropped (normal prior)
    # classify
    per_particle_snr: float = 1.0
    cluster_components: int = 5
    cluster_method: str = 'leiden'                # 'leiden', 'kmeans' or 'hdbscan'
    min_cluster_size: int = 10                    # smallest seed cluster that gets a reference
    n_polish_rounds: int = 2
    shift_range_px: int = 3                       # shift search half-width, working pixels
    bin_factor: int = 2                           # alignment runs on boxes binned by this factor
    merge_fraction: float = 1.0                   # fraction of the higher half-split ceiling two references must reach to merge
    reference_quantile: float = 0.99              # null quantile a reference's median correlation must clear
    reject_quantile: float = 0.95                 # null quantile of the best correlation below which a particle is rejected
    # identify
    seed: int = 0
