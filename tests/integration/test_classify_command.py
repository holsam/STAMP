'''
STAMP: integration tests for `stamp classify` command
'''

# Import external dependencies
import json, mrcfile, numpy as np
from pathlib import Path
from typer.testing import CliRunner

# Import main CLI
from stamp.classify.extract import extract_particle_set
from stamp.classify.features import build_feature_matrix
from stamp.classify.align import align_inplane
from stamp.classify.average import compute_class_averages
from stamp.cli.cli import stamp_app
from stamp.schemas.particles import HalfSet, Particle

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
        result = runner.invoke(
            stamp_app,
            [
                '-d', str(tmp_path),
                'classify',
                '--particles', str(particle_set_path),
                '--raw-dir', str(raw_dir),
                '--out-dir', tmp_path,
                '--voxel-size-a', '10.0',
                '--box-length-a', '200.0',
                '--method', 'kmeans',
                '--n-clusters', '2',
                '--n-components', '10',
            ],
        )
        output_dir = tmp_path / 'stamp' / 'classify'
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
            stamp_app,
            [
                '-vv',
                '-d', str(tmp_path),
                'classify',
                '--particles', str(particle_set_path),
                '--raw-dir', str(raw_dir),
                '--out-dir', tmp_path,
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

class TestClassifyChunking:
    def _guard_no_full_concatenate(self, monkeypatch, max_rows: int) -> None:
        '''Fail the test if anything tries to stack more than max_rows at once.'''
        real_stack = np.stack
        def guarded_stack(arrays, *args, **kwargs):
            if len(arrays) > max_rows:
                raise AssertionError(f'np.stack called with {len(arrays)} arrays, exceeding the per-batch bound of {max_rows}')
            return real_stack(arrays, *args, **kwargs)
        monkeypatch.setattr(np, 'stack', guarded_stack)

    def test_classify_pipeline_never_stacks_the_whole_dataset(self, tmp_path, monkeypatch):
        '''extract/features/align/average batch per-tomogram, instead of stacking the whole dataset.'''
        n_tomograms = 4
        n_particles_per_tomogram = 20
        box_voxels = 9
        rng = np.random.default_rng(0)
        raw_dir = tmp_path / 'raw'
        raw_dir.mkdir()
        tomogram_paths: dict[str, str] = {}
        particles: list[Particle] = []
        grid = [(10 + (i % 5) * 5, 10 + (i // 5) * 5, 20) for i in range(n_particles_per_tomogram)]
        for t in range(n_tomograms):
            tomogram_id = f'tomo{t:03d}'
            volume = rng.normal(0.0, 0.2, size=(40, 40, 40)).astype(np.float32)
            for x, y, z in grid:
                volume[z - 2 : z + 2, y - 2 : y + 2, x - 2 : x + 2] -= 3.0
            path = raw_dir / f'{tomogram_id}.mrc'
            with mrcfile.new(path, overwrite=True) as mrc:
                mrc.set_data(volume)
            tomogram_paths[tomogram_id] = str(path)
            for index, (x, y, z) in enumerate(grid):
                particles.append(
                    Particle(
                        particle_id=f'{tomogram_id}_p{index:03d}',
                        tomogram_id=tomogram_id,
                        position=(float(x), float(y), float(z)),
                        orientation=(1.0, 0.0, 0.0, 0.0),
                        source_picker='stamp-native',
                        half_set=HalfSet.A if index % 2 == 0 else HalfSet.B,
                    )
                )
        cache_dir = tmp_path / 'subvolumes'
        self._guard_no_full_concatenate(monkeypatch, max_rows=n_particles_per_tomogram)
        subvols, kept, skipped = extract_particle_set(particles, tomogram_paths, box_voxels, cache_dir=cache_dir)
        assert not skipped
        assert len(subvols) == n_tomograms * n_particles_per_tomogram
        monkeypatch.undo()
        features = build_feature_matrix(subvols, n_workers=1, batch_size=n_particles_per_tomogram)
        assert features.shape[0] == len(subvols)
        cluster_ids = ['c00'] * len(subvols)
        angles = align_inplane(subvols, cluster_ids, n_workers=1)
        assert len(angles) == len(subvols)
        averages = compute_class_averages(subvols, cluster_ids)
        assert averages['c00'][1] == len(subvols)