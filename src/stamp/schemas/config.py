'''
STAMP: stamp_run.toml schema for `stamp run`
'''

# Import external dependencies
import tomllib
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.utils.errors import StampPipelineError
from stamp.utils.log import log, log_config

# _Strict: reject unknown keys, treat "" as unset
class _Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='before')
    @classmethod
    def _blank_to_none(cls, data: object) -> object:
        if isinstance(data, dict):
            return {key: (None if value == '' else value) for key, value in data.items()}
        return data

# RunSettings: the [run] table
class RunSettings(_Strict):
    segmentation_dir: Path | None = None    # required for the normal orientation prior only
    raw_tomogram_dir: Path
    output_dir: Path
    voxel_size_angstrom: float | None = None
    backend: Literal['local', 'mock', 'cluster'] = 'local'
    stop_after: Literal['pick', 'classify', 'identify', 'refine'] | None = None
    orientation_prior: OrientationPrior = OrientationPrior.NORMAL
    tilt_max_degrees: float | None = Field(default=None, gt=0, lt=90)   # None means no missing wedge is modelled
    tilt_axis_degrees: float = 90.0     # angle of the tilt axis from the tomogram x axis in the xy plane, 90 is a tilt about y

# PlotSettings: the [plots] table
class PlotSettings(_Strict):
    enabled: bool = True
    format: Literal['png', 'jpg', 'tiff', 'svg'] = 'tiff'
    pick_style: Literal['segmented', 'none'] = 'segmented'
    pick_zstack_movie: bool = False
    pick_plot_3d: bool = False

# DecoySettings: the [decoy] table
class DecoySettings(_Strict):
    enabled: bool = True
    method: Literal['shifted', 'synthetic-noise', 'random-volume'] = 'shifted'
    n_decoys_per_tomogram: int | None = None    # synthetic-noise only
    min_distance_from_real_angstrom: float = 100.0
    min_distance_from_picks_angstrom: float = 60.0
    min_shift_angstrom: float = 200.0
    max_shift_angstrom: float = 600.0
    min_border_angstrom: float = 200.0
    n_synthetic_tomograms: int = 3
    synthetic_shape_voxels: tuple[int, int, int] | None = None  # None = match the first real tomogram
    n_workers: int = 1
    keep_raw: bool = False

# Per-stage tables mirror each command's options
class PickStage(_Strict):
    picks_files: list[Path]
    picker_names: list[str] | None = None  # defaults to file stems
    consensus_rule: Literal['union', 'intersection'] = 'intersection'
    distance_threshold: float = 15.0
    half_set_seed: int = 0
    max_surface_distance_angstrom: float = 300.0
    backend: Literal['local', 'mock', 'cluster'] | None = None  # inert for pick, kept so run.backend inheritance is uniform
    n_workers: int = 1
    keep_raw: bool = False

class ClassifyStage(_Strict):
    method: Literal['leiden', 'hdbscan', 'kmeans'] = 'leiden'
    min_cluster_size: int = 20
    n_clusters: int = 5
    n_components: int = 20
    box_angstrom: float = 300.0
    n_radial_bins: int = 12
    strict_halfset_independence: bool = True
    inplane_angular_step_degrees: float = 10.0
    bin_factor: int = Field(default=2, ge=1)
    n_rotations: int = Field(default=3000, ge=2)
    shift_range_voxels: int = Field(default=3, ge=0)
    polish_rounds: int = Field(default=2, ge=1)
    null_sigmas: float = Field(default=3.0, ge=0)
    reject_quantile: float = Field(default=0.95, gt=0.5, lt=1.0)
    merge_fraction: float = Field(default=1.0, ge=0)
    random_state: int = 0
    n_workers: int = 1
    keep_raw: bool = False

class IdentifyStage(_Strict):
    candidates: Path
    resolution: float | None = None
    fitter: Literal['native'] = 'native'
    fetch_missing: bool = False
    backend: Literal['local', 'mock', 'cluster'] | None = None
    n_workers: int = 1

class RefineStage(_Strict):
    tool: Literal['relion', 'm'] = 'relion'
    class_id: str = 'all'
    threshold: float = 0.143
    iterations: int = 5
    mask: Path | None = None
    backend: Literal['local', 'mock', 'cluster'] | None = None

# StageConfigs: the [stage.*] tables
class StageConfigs(_Strict):
    pick: PickStage
    classify: ClassifyStage = ClassifyStage()
    identify: IdentifyStage
    refine: RefineStage = RefineStage()

# RunConfig: the whole stamp_run.toml
class RunConfig(_Strict):
    run: RunSettings
    decoy: DecoySettings = DecoySettings()
    plots: PlotSettings = PlotSettings()
    stage: StageConfigs

    @model_validator(mode='after')
    def _cross_field(self) -> 'RunConfig':
        if self.run.orientation_prior is OrientationPrior.NORMAL and self.run.segmentation_dir is None:
            raise StampPipelineError('[run].segmentation_dir is required for orientation_prior "normal", orientations are inferred from the membrane segmentation')
        if self.run.orientation_prior is not OrientationPrior.NORMAL and self.decoy.enabled and self.decoy.method != 'random-volume':
            raise StampPipelineError(f'[decoy].method "{self.decoy.method}" is membrane-based, use "random-volume" (or disable the decoy) for orientation_prior "{self.run.orientation_prior.value}"')
        stop = self.run.stop_after
        needs_identify = stop in (None, 'identify', 'refine')
        if needs_identify and not self.stage.identify.candidates:
            raise StampPipelineError('[stage.identify].candidates is required unless stop_after is "pick" or "classify"')
        if needs_identify and self.stage.identify.resolution is None:
            raise StampPipelineError('[stage.identify].resolution is required (Å) unless stop_after is "pick" or "classify"')
        return self

# load_run_config: parse and validate stamp_run.toml, resolving paths relative to the file
def load_run_config(path: Path) -> RunConfig:
    raw = tomllib.loads(path.read_text())
    config = RunConfig.model_validate(raw)
    base = path.parent
    if config.run.segmentation_dir is not None:
        config.run.segmentation_dir = (base / config.run.segmentation_dir).resolve()
    config.run.raw_tomogram_dir = (base / config.run.raw_tomogram_dir).resolve()
    config.run.output_dir = (base / config.run.output_dir).resolve()
    config.stage.identify.candidates = (base / config.stage.identify.candidates).resolve()
    if config.stage.refine.mask is not None:
        config.stage.refine.mask = (base / config.stage.refine.mask).resolve()
    config.stage.pick.picks_files = [(base / path).resolve() for path in config.stage.pick.picks_files]
    if config.stage.pick.backend is None:
        log.debug(f'stage.pick.backend not set, inheriting run.backend={config.run.backend!r}')
        config.stage.pick.backend = config.run.backend
    if config.stage.identify.backend is None:
        log.debug(f'stage.identify.backend not set, inheriting run.backend={config.run.backend!r}')
        config.stage.identify.backend = config.run.backend
    if config.stage.refine.backend is None:
        log.debug(f'stage.refine.backend not set, inheriting run.backend={config.run.backend!r}')
        config.stage.refine.backend = config.run.backend
    log_config('run', config)
    return config
