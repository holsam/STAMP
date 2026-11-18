'''
STAMP: stream directory layout, fixed settings and per-tomogram completion
'''

# Import external dependencies
import json
from pathlib import Path
from pydantic import BaseModel

# Import internal STAMP objects
from stamp.utils.checkpoint import atomic_write_json
from stamp.utils.errors import StampValidationError

# StreamSettings: settings that affect results
class StreamSettings(BaseModel):
    voxel_size_angstrom: float
    box_voxels: int
    picker_names: list[str]
    consensus_rule: str
    distance_threshold: float
    orientation_prior: str
    max_surface_distance_angstrom: float
    max_beam_angle_deviation: float | None
    decoy_method: str
    n_decoys_per_tomogram: int | None = None     # recorded only if a decoy option needs it
    min_shift_angstrom: float
    max_shift_angstrom: float
    min_distance_from_real_angstrom: float
    min_distance_from_picks_angstrom: float
    min_border_angstrom: float
    density_sign: int
    seed: int
    half_set_seed: int
    finalised: bool = False

# StreamDir: paths inside one stream directory
class StreamDir:
    # __init__: bind stream directory root
    def __init__(self, root: Path) -> None:
        self.root = root

    # settings_path: stream.json, holding the fixed StreamSettings
    @property
    def settings_path(self) -> Path:
        return self.root / 'stream.json'

    # tomogram_dir: per-tomogram output directory
    def tomogram_dir(self, tomogram_id: str) -> Path:
        return self.root / 'tomograms' / tomogram_id

    # particle_dir: per-particle MRC export directory for one tomogram
    def particle_dir(self, tomogram_id: str) -> Path:
        return self.root / 'particles' / tomogram_id

    # subvolume_dir: subvolume cache directory for one kind and prior
    def subvolume_dir(self, kind: str, prior: str) -> Path:     # kind is 'real' or 'decoy'
        return self.root / 'subvolumes' / kind / prior

    # is_done: True when tomogram's done.json checkpoint exists
    def is_done(self, tomogram_id: str) -> bool:
        return (self.tomogram_dir(tomogram_id) / 'done.json').is_file()

    # done_tomograms: sorted ids of every completed tomogram
    def done_tomograms(self) -> list[str]:
        return sorted(path.parent.name for path in (self.root / 'tomograms').glob('*/done.json'))

# load_or_init_settings: read stream.json, or write it on the first append
def load_or_init_settings(stream: StreamDir, requested: StreamSettings) -> StreamSettings:
    if not stream.settings_path.is_file():
        stream.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stream.settings_path, requested.model_dump())
        return requested
    stored = StreamSettings.model_validate(json.loads(stream.settings_path.read_text()))
    if stored.finalised:
        raise StampValidationError(f'{stream.root} is already finalised, start a new stream directory to add tomograms')
    differing = sorted(key for key, value in requested.model_dump().items() if key != 'finalised' and value != getattr(stored, key))
    if differing:
        raise StampValidationError(f'{stream.root} was started with different settings for {differing}; the first append fixes them, start a new stream directory to change them')
    return stored
