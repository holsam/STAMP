'''
STAMP: stamp_run.toml schema for `stamp run`
'''

# Import external dependencies
import tomllib
from pathlib import Path
from pydantic import BaseModel, ConfigDict, model_validator
from typing import Literal

# Import internal STAMP objects
from stamp.utils.errors import StampPipelineError
from stamp.utils.log import log

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
    segmentation_dir: Path
    raw_tomogram_dir: Path
    output_dir: Path
    voxel_size_angstrom: float | None = None
    backend: Literal['local', 'mock', 'cluster'] = 'local'
    stop_after: Literal['pick', 'classify', 'identify', 'refine'] | None = None

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
    method: Literal['rejected-surface', 'shifted', 'synthetic-noise'] = 'rejected-surface'
    n_decoys_per_tomogram: int | None = None    # None = adaptive (target same N as real picks)
    min_distance_from_real_angstrom: float = 100.0
    min_distance_from_picks_angstrom: float = 60.0
    min_shift_angstrom: float = 200.0
    max_shift_angstrom: float = 600.0
    n_synthetic_tomograms: int = 3
    synthetic_shape_voxels: tuple[int, int, int] | None = None  # None = match the first real tomogram
    n_workers: int = 1
    keep_raw: bool = False

# Per-stage tables mirror each command's options
class PickStage(_Strict):
    pickers: list[str] = ['stamp-native']
    consensus_rule: Literal['union', 'intersection'] = 'intersection'
    distance_threshold: float = 15.0
    half_set_seed: int = 0
    backend: Literal['local', 'mock', 'cluster'] | None = None
    n_workers: int = 1
    keep_raw: bool = False

class ClassifyStage(_Strict):
    method: Literal['hdbscan', 'kmeans'] = 'hdbscan'
    min_cluster_size: int = 20
    n_clusters: int = 5
    n_components: int = 20
    box_angstrom: float = 300.0
    n_radial_bins: int = 12
    strict_halfset_independence: bool = True
    inplane_alignment: bool = True
    inplane_angular_step_degrees: float = 10.0
    inplane_iterations: int = 3
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
    pick: PickStage = PickStage()
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
    config.run.segmentation_dir = (base / config.run.segmentation_dir).resolve()
    config.run.raw_tomogram_dir = (base / config.run.raw_tomogram_dir).resolve()
    config.run.output_dir = (base / config.run.output_dir).resolve()
    config.stage.identify.candidates = (base / config.stage.identify.candidates).resolve()
    if config.stage.refine.mask is not None:
        config.stage.refine.mask = (base / config.stage.refine.mask).resolve()
    if config.stage.pick.backend is None:
        log.debug(f'stage.pick.backend not set, inheriting run.backend={config.run.backend!r}')
        config.stage.pick.backend = config.run.backend
    if config.stage.identify.backend is None:
        log.debug(f'stage.identify.backend not set, inheriting run.backend={config.run.backend!r}')
        config.stage.identify.backend = config.run.backend
    if config.stage.refine.backend is None:
        log.debug(f'stage.refine.backend not set, inheriting run.backend={config.run.backend!r}')
        config.stage.refine.backend = config.run.backend
    return config
