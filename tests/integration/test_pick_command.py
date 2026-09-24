'''
STAMP: integration tests for `stamp pick` command
'''

# Import external dependencies
import json, mrcfile, numpy as np
from pathlib import Path
from typer.testing import CliRunner

# Import internal STAMP objects
from stamp.cli.cli import stamp_app
from stamp.schemas.particles import ParticleSet

# Initialise runner
runner = CliRunner()

# _write_pair: write a vesicle with one dark particle as a matched segmentation/tomogram pair
# in _write_pair, add a keyword-only voxel_size param:
def _write_pair(seg_dir: Path, raw_dir: Path, tomogram_id: str, *, voxel_size: float | None = None) -> None:
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
        if voxel_size is not None:
            mrc.voxel_size = voxel_size

# TestPickCommand: class containing tests for pick command
class TestPickCommand:
    # check `stamp pick` runs the native picker and writes outputs
    def test_pick_runs_native_picker_end_to_end(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'tomo000')
        result = runner.invoke(
            stamp_app,
            [
                '-d', str(tmp_path),
                'pick',
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', tmp_path,
                '--voxel-size-a', '10.0',
                '--consensus-rule', 'union',
                '--picker-params', json.dumps({'stamp-native': {'n_mad': 2.5, 'offset_windows_angstrom': [[20.0, 80.0]]}}),
                '--backend', 'local',
                'stamp-native',
            ],
        )

        assert result.exit_code == 0, result.output
        output_dir = tmp_path / 'stamp' / 'pick'
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
            stamp_app,
            [
                '-d', str(tmp_path),
                'pick',
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', tmp_path,
                '--voxel-size-a', '10.0',
                '--picker-params', json.dumps({'stamp-native': {'n_mad': 2.5, 'offset_windows_angstrom': [[20.0, 80.0]]}}),
                '--backend', 'local',
                'stamp-native',
            ],
        )
        assert 'No raw tomogram matching orphan, skipping' in result.output
        assert result.exit_code == 0, result.output

    # check a segmentation whose stem is the raw stem plus a suffix still matches
    def test_pick_matches_segmentation_by_stem_prefix(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        seg_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        shape = (60, 60, 60)
        centre = np.array(shape) / 2.0
        grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1)
        distance = np.linalg.norm(grid - centre, axis=-1)
        segmentation = ((distance > 16.0) & (distance < 20.0)).astype(np.float32)
        tomogram = np.zeros(shape, dtype=np.float32)
        tomogram[segmentation > 0] = -1.0
        tomogram[42:47, 28:33, 28:33] = -8.0
        with mrcfile.new(raw_dir / 'Position_1_stack_Vol.mrc', overwrite=True) as mrc:
            mrc.set_data(tomogram)
        with mrcfile.new(seg_dir / 'Position_1_stack_Vol.denoised_segmented.mrc', overwrite=True) as mrc:
            mrc.set_data(segmentation)
        result = runner.invoke(
            stamp_app,
            [
                '-d', str(tmp_path),
                'pick',
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', tmp_path,
                '--voxel-size-a', '10.0',
                '--picker-params', json.dumps({'stamp-native': {'n_mad': 2.5, 'offset_windows_angstrom': [[20.0, 80.0]]}}),
                '--backend', 'local',
                'stamp-native',
            ],
        )
        assert result.exit_code == 0, result.output
        assert 'No raw tomogram matching' not in result.output

    # check voxel size is read from the raw tomogram's MRC header when --voxel-size-a is omitted
    def test_pick_reads_voxel_size_from_header_when_omitted(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'tomo000', voxel_size=10.0)
        result = runner.invoke(
            stamp_app,
            [
                '-d', str(tmp_path),
                'pick',
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', tmp_path,
                '--picker-params', json.dumps({'stamp-native': {'n_mad': 2.5, 'offset_windows_angstrom': [[20.0, 80.0]]}}),
                '--backend', 'local',
                'stamp-native',
            ],
        )
        assert result.exit_code == 0, result.output

    # check mixed voxel sizes across tomograms warn and fall back to the most common
    def test_pick_warns_and_uses_most_common_voxel_size(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'tomo000', voxel_size=10.0)
        _write_pair(seg_dir, raw_dir, 'tomo001', voxel_size=10.0)
        _write_pair(seg_dir, raw_dir, 'tomo002', voxel_size=14.0)
        result = runner.invoke(
            stamp_app,
            [
                '-d', str(tmp_path),
                'pick',
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', tmp_path,
                '--picker-params', json.dumps({'stamp-native': {'n_mad': 2.5, 'offset_windows_angstrom': [[20.0, 80.0]]}}),
                '--backend', 'local',
                'stamp-native',
            ],
        )
        assert result.exit_code == 0, result.output
        assert 'Multiple voxel sizes were found' in result.output