'''
STAMP: configuration dataclasses for the workflow diagrams
'''

# Import external dependencies
from dataclasses import dataclass, field
from pathlib import Path

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
    field_px: int = 760                           # side of the mock tomogram slice, pixels
    # scene composition
    n_vesicles: int = 10
    n_target: int = 75
    n_distractor: int = 60
    n_extra: int = 32                             # particles per extra / unknown class
    # classify
    per_particle_snr: float = 0.3
    cluster_components: int = 10
    kmeans_k: int | None = None                   # KMeans clusters (default: n species + 1)
    hdbscan_min_cluster_size: int = 15
    # identify
    identify_min_cc: float = 0.72
    identify_min_gap: float = 0.06
    decoy_margin: float = 0.05
    n_decoy_trials: int = 40
    seed: int = 0
