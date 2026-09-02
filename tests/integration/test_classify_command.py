'''
STAMP: integration tests for `stamp classify` command
'''

# Import external dependencies
import json, mrcfile, numpy as np
from pathlib import Path
from typer.testing import CliRunner

# Import main CLI
from stamp.cli.cli import stamp

# Initialise runner
runner = CliRunner()

# _build_dataset: a tomogram with 60 'large' and 60 'small' particles on a flat membrane
def _build_dataset(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    shape = (80, 200, 200)
    volume = rng.normal(0.0, 0.2, size=shape).astype(np.float32)

    particles = []
    for index in range(120):
        x = 20 + (index % 12) * 15
        y = 20 + (index // 12) * 15
        z = 40
        radius = 4 if index < 60 else 2
        volume[
            z + 4 - radius : z + 4 + radius,
            y - radius : y + radius,
            x - radius : x + radius,
        ] -= 3.0
        particles.append(
            {
                'particle_id': f'p{index:06d}',
                'tomogram_id': 'tomo000',
                'position': [float(x), float(y), float(z)],
                'orientation': [1.0, 0.0, 0.0, 0.0],
                'source_picker': 'stamp-native',
                'confidence': 5.0,
                'half_set': 'A' if index % 2 == 0 else 'B',
            }
        )

    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    with mrcfile.new(raw_dir / 'tomo000.mrc', overwrite=True) as mrc:
        mrc.set_data(volume)

    particle_set_path = tmp_path / 'particle_set.json'
    particle_set_path.write_text(
        json.dumps(
            {
                'particles': particles,
                'consensus_rule': 'union',
                'contributing_pickers': ['stamp-native'],
            }
        )
    )
    return particle_set_path, raw_dir

# TestClassifyCommand: class containing unit tests for test_classify_command.py
class TestClassifyCommand:
    def test_classify_end_to_end(self, tmp_path: Path) -> None:
        '''Classify runs end to end and writes assignments, averages and params.'''
        particle_set_path, raw_dir = _build_dataset(tmp_path)
        output_dir = tmp_path / 'classify'
        result = runner.invoke(
            stamp,
            [
                'classify',
                '--particles', str(particle_set_path),
                '--raw-dir', str(raw_dir),
                '--out-dir', str(output_dir),
                '--voxel-size-a', '10.0',
                '--box-length-a', '200.0',
                '--method', 'kmeans',
                '--n-clusters', '2',
                '--n-components', '10',
            ],
        )
        assert result.exit_code == 0, result.output
        assignments = json.loads((output_dir / 'class_assignments.json').read_text())
        assert len(assignments) > 0
        assert {a['cluster_id'] for a in assignments} == {'c00', 'c01'}
        averages = list((output_dir / 'class_averages').glob('*.mrc'))
        assert averages, 'no class averages written'
        # Averages are written per half-set.
        assert any('halfA' in path.name for path in averages)
        assert any('halfB' in path.name for path in averages)
        assert (output_dir / 'params.toml').exists()

    def test_classify_strict_halfset_mode(self, tmp_path: Path) -> None:
        '''Strict mode reports cross-half cluster matching.'''
        particle_set_path, raw_dir = _build_dataset(tmp_path)
        result = runner.invoke(
            stamp,
            [
                'classify',
                '--particles', str(particle_set_path),
                '--raw-dir', str(raw_dir),
                '--out-dir', str(tmp_path / 'strict'),
                '--voxel-size-a', '10.0',
                '--box-length-a', '200.0',
                '--method', 'kmeans',
                '--n-clusters', '2',
                '--n-components', '10',
                '--strict-halfset-independence',
            ],
        )
        assert result.exit_code == 0, result.output
        assert 'Cross-half cluster matching' in result.output
