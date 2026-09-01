'''
STAMP: integration tests for `stamp pick` command
'''

# Import external dependencies
import json, mrcfile, numpy as np
from pathlib import Path
from typer.testing import CliRunner

# Import internal STAMP objects
from stamp.cli.cli import stamp
from stamp.schemas.particles import ParticleSet

# Initialise runner
runner = CliRunner()

# _write_pair: write a vesicle with one dark particle as a matched segmentation/tomogram pair
def _write_pair(seg_dir: Path, raw_dir: Path, tomogram_id: str) -> None:
    shape = (60, 60, 60)
    centre = np.array(shape) / 2.0
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1)
    distance = np.linalg.norm(grid - centre, axis=-1)

    segmentation = ((distance > 16.0) & (distance < 20.0)).astype(np.float32)
    tomogram = np.zeros(shape, dtype=np.float32)
    tomogram[segmentation > 0] = -1.0
    tomogram[42:47, 28:33, 28:33] = -8.0

    seg_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(seg_dir / f'{tomogram_id}.mrc', overwrite=True) as mrc:
        mrc.set_data(segmentation)
    with mrcfile.new(raw_dir / f'{tomogram_id}.mrc', overwrite=True) as mrc:
        mrc.set_data(tomogram)

# TestPickCommand: class containing tests for pick command
class TestPickCommand:
    # check `stamp pick` runs the native picker and writes outputs
    def test_pick_runs_native_picker_end_to_end(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        output_dir = tmp_path / 'out'
        _write_pair(seg_dir, raw_dir, 'tomo000')

        result = runner.invoke(
            stamp,
            [
                'pick',
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', str(output_dir),
                '--voxel-size-a', '10.0',
                '--consensus-rule', 'union',
                '--picker-params', json.dumps({'stamp-native': {'n_mad': 2.5}}),
                '--backend', 'local',
                'stamp-native',
            ],
        )

        assert result.exit_code == 0, result.output
        particle_set_path = output_dir / 'particle_set.json'
        assert particle_set_path.exists()

        particle_set = ParticleSet.model_validate(json.loads(particle_set_path.read_text()))
        assert particle_set.particles
        assert all(p.half_set is not None for p in particle_set.particles)
        assert all(p.orientation is not None for p in particle_set.particles)
        assert (output_dir / 'params.toml').exists()

    # check a segmentation with no raw match is skipped
    def test_pick_skips_segmentations_without_raw_match(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'tomo000')
        # an extra segmentation with no matching raw tomogram.
        with mrcfile.new(seg_dir / 'orphan.mrc', overwrite=True) as mrc:
            mrc.set_data(np.zeros((60, 60, 60), dtype=np.float32))

        result = runner.invoke(
            stamp,
            [
                'pick',
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', str(tmp_path / 'out'),
                '--voxel-size-a', '10.0',
                '--picker-params', json.dumps({'stamp-native': {'n_mad': 2.5}}),
                '--backend', 'local',
                'stamp-native',
            ],
        )
        assert "no raw tomogram matching 'orphan'" in result.output
        assert result.exit_code == 0, result.output
