'''
STAMP: unit tests for stream state and the settings compatibility check
'''

# Import external dependencies
import json, pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.stream.state import StreamDir, StreamSettings, load_or_init_settings
from stamp.utils.errors import StampValidationError

# _settings: a StreamSettings with defaults overridable per test
def _settings(**overrides) -> StreamSettings:
    values = dict(
        voxel_size_angstrom=10.0, box_voxels=32, picker_names=['a', 'b'], consensus_rule='union',
        distance_threshold=40.0, orientation_prior='normal', max_surface_distance_angstrom=300.0,
        max_beam_angle_deviation=None, decoy_method='auto', min_shift_angstrom=100.0, max_shift_angstrom=200.0,
        min_distance_from_real_angstrom=50.0, min_distance_from_picks_angstrom=60.0, min_border_angstrom=100.0,
        density_sign=1, seed=0, half_set_seed=0,
    )
    values.update(overrides)
    return StreamSettings(**values)

# TestStreamSettings: class checking that first append writes stream.json and later appends match it
class TestStreamSettings:
    def test_first_call_writes_settings(self, tmp_path: Path) -> None:
        stream = StreamDir(tmp_path / 'stream')
        load_or_init_settings(stream, _settings())
        assert json.loads(stream.settings_path.read_text())['box_voxels'] == 32

    def test_matching_second_call_passes(self, tmp_path: Path) -> None:
        stream = StreamDir(tmp_path / 'stream')
        load_or_init_settings(stream, _settings())
        assert load_or_init_settings(stream, _settings()) == _settings()

    @pytest.mark.parametrize('override, key', [
        (dict(box_voxels=40), 'box_voxels'),
        (dict(picker_names=['a', 'c']), 'picker_names'),
        (dict(seed=1), 'seed'),
    ])
    def test_differing_setting_is_rejected_naming_the_key(self, tmp_path: Path, override: dict, key: str) -> None:
        stream = StreamDir(tmp_path / 'stream')
        load_or_init_settings(stream, _settings())
        with pytest.raises(StampValidationError, match=key):
            load_or_init_settings(stream, _settings(**override))

    def test_finalised_stream_rejects_append(self, tmp_path: Path) -> None:
        stream = StreamDir(tmp_path / 'stream')
        load_or_init_settings(stream, _settings(finalised=True))
        with pytest.raises(StampValidationError, match='finalised'):
            load_or_init_settings(stream, _settings())
