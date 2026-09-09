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
    # native picker
    offset_min_a: float = 20.0
    offset_max_a: float = 80.0
    n_samples: int = 5
    surface_spacing_a: float = 20.0
    min_particle_distance_a: float = 60.0
    n_mad: float = 3.0
    density_sign: int = 1
    seed: int = 0

# NativePickerConfig: native-picker parameters for the mock, mirroring stamp.picking.native.NativePickerConfig
@dataclass(frozen=True)
class NativePickerConfig:
    pixel_size_angstrom: float
    offset_min_angstrom: float = 20.0
    offset_max_angstrom: float = 80.0
    n_samples: int = 5
    surface_spacing_angstrom: float = 20.0
    min_particle_distance_angstrom: float = 60.0
    n_mad: float = 3.0
    density_sign: int = 1

    # to_voxels: convert a distance in Angstrom to voxels
    def to_voxels(self, angstrom: float) -> float:
        return angstrom / self.pixel_size_angstrom

    # from_diagram_config: pull the native-picker knobs out of a DiagramConfig
    @classmethod
    def from_diagram_config(cls, config: DiagramConfig) -> 'NativePickerConfig':
        return cls(
            pixel_size_angstrom=config.pixel_size,
            offset_min_angstrom=config.offset_min_a,
            offset_max_angstrom=config.offset_max_a,
            n_samples=config.n_samples,
            surface_spacing_angstrom=config.surface_spacing_a,
            min_particle_distance_angstrom=config.min_particle_distance_a,
            n_mad=config.n_mad,
            density_sign=config.density_sign,
        )
