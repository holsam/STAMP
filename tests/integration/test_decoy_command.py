'''
STAMP: integration tests of `stamp decoy` for all three methods
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

# _write_pair: write a matched segmentation/tomogram vesicle pair into the given directories
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

# _run_pick: run `stamp pick` on a fresh vesicle pair and return its paths
def _run_pick(tmp_path: Path) -> tuple[Path, Path, Path]:
    seg_dir, raw_dir, pick_out = tmp_path / 'seg', tmp_path / 'raw', tmp_path / 'pick'
    _write_pair(seg_dir, raw_dir, 'tomo000')
    result = runner.invoke(
        stamp,
        [
            'pick',
            '--seg-dir', str(seg_dir),
            '--raw-dir', str(raw_dir),
            '--out-dir', str(pick_out),
            '--voxel-size-a', '10.0',
            '--picker-params', json.dumps({'n_mad': 2.5}),
            '--backend', 'local',
            'stamp-native'
        ],
    )
    assert result.exit_code == 0, result.output
    return seg_dir, raw_dir, pick_out / 'particle_set.json'

# TestDecoyCommand: class containing unit tests for test_decoy_command.py
class TestDecoyCommand:
    def test_decoy_rejected_surface_end_to_end(self, tmp_path: Path) -> None:
        '''Rejected-surface runs end to end and writes a decoy set.'''
        seg_dir, raw_dir, particle_set_path = _run_pick(tmp_path)
        output_dir = tmp_path / 'decoy'

        result = runner.invoke(
            stamp,
            [
                'decoy',
                '--method', 'rejected-surface',
                '--real-particle-set', str(particle_set_path),
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', str(output_dir),
                '--voxel-size-a', '10.0',
                '--picker-params', json.dumps({'n_mad': 2.5}),
                '--n-decoys-per-tomogram', '20',
                '--seed', '1',
            ],
        )
        assert result.exit_code == 0, result.output

        decoy_set = ParticleSet.model_validate(
            json.loads((output_dir / 'decoy_particle_set.json').read_text())
        )
        assert decoy_set.particles
        assert all(p.source_picker.startswith('decoy-') for p in decoy_set.particles)

    def test_decoy_shifted_end_to_end(self, tmp_path: Path) -> None:
        '''Shifted runs end to end without error.'''
        seg_dir, raw_dir, particle_set_path = _run_pick(tmp_path)
        result = runner.invoke(
            stamp,
            [
                'decoy',
                '--method', 'shifted',
                '--real-particle-set', str(particle_set_path),
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', str(tmp_path / 'decoy_shifted'),
                '--voxel-size-a', '10.0',
                '--min-shift-a', '100.0',
                '--max-shift-a', '180.0',
                '--seed', '2',
            ],
        )
        assert result.exit_code == 0, result.output

    def test_decoy_synthetic_noise_end_to_end(self, tmp_path: Path) -> None:
        '''Synthetic-noise writes the decoy set, manifests and volume dirs.'''
        output_dir = tmp_path / 'decoy_noise'
        result = runner.invoke(
            stamp,
            [
                'decoy',
                '--method', 'synthetic-noise',
                '--out-dir', str(output_dir),
                '--voxel-size-a', '10.0',
                '--n-synthetic-tomograms', '2',
                '--synthetic-shape', '40,40,40',
                '--n-decoys-per-tomogram', '5',
                '--seed', '3',
            ],
        )
        assert result.exit_code == 0, result.output
        assert (output_dir / 'decoy_particle_set.json').exists()
        assert (output_dir / 'decoy_manifests.json').exists()
        assert (output_dir / 'segmentations').is_dir()
        assert (output_dir / 'tomograms').is_dir()
