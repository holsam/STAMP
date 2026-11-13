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

class TestClassifyChunking:
    def _guard_no_full_concatenate(self, monkeypatch, max_rows: int) -> None:
        '''Fail the test if anything tries to stack more than max_rows at once.'''
        real_stack = np.stack
        def guarded_stack(arrays, *args, **kwargs):
            if len(arrays) > max_rows:
                raise AssertionError(f'np.stack called with {len(arrays)} arrays, exceeding the per-batch bound of {max_rows}')
            return real_stack(arrays, *args, **kwargs)
        monkeypatch.setattr(np, 'stack', guarded_stack)

    def test_classify_resumes_from_matching_partial_output(self, tmp_path: Path) -> None:
        '''A second identical run reuses extraction, features and alignment; a changed command starts fresh.'''
        particle_set_path, raw_dir = _build_dataset(tmp_path)
        argv = [
            '-vv', '-d', str(tmp_path), 'classify',
            '--particles', str(particle_set_path), '--raw-dir', str(raw_dir), '--out-dir', str(tmp_path),
            '--voxel-size-a', '10.0', '--box-length-a', '200.0', '--method', 'kmeans',
            '--n-clusters', '2', '--n-components', '10', '--keep-raw', '--no-strict-halfset-independence',
        ]
        assert runner.invoke(stamp_app, argv).exit_code == 0
        output_dir = tmp_path / 'stamp' / 'classify'
        first = (output_dir / 'class_assignments.json').read_text()

        result = runner.invoke(stamp_app, argv)
        assert result.exit_code == 0, result.output
        assert 'Resuming extraction' in result.output
        assert 'features loaded from checkpoint' in result.output
        assert 'Resuming polish step' in result.output
        assert (output_dir / 'class_assignments.json').read_text() == first

        changed = argv[:argv.index('--n-clusters')] + ['--n-clusters', '3'] + argv[argv.index('--n-clusters') + 2:]
        result = runner.invoke(stamp_app, changed)
        assert result.exit_code == 0, result.output
        assert 'starting fresh' in result.output
        assert 'Resuming extraction' not in result.output

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

class TestClassifyCommand:
    def _invoke(self, tmp_path: Path, *extra: str, edit=None):
        particle_set_path, raw_dir = _build_dataset(tmp_path)
        if edit is not None:
            payload = json.loads(particle_set_path.read_text())
            edit(payload)
            particle_set_path.write_text(json.dumps(payload))
        return runner.invoke(stamp_app, [
            '-vv', '-d', str(tmp_path), 'classify', '--particles', str(particle_set_path), '--raw-dir', str(raw_dir), '--out-dir', str(tmp_path),
            '--voxel-size-a', '10.0', '--box-length-a', '200.0', '--method', 'kmeans', '--n-clusters', '2', '--n-components', '10',
            '--min-cluster-size', '10', '--tilt-max', '60', *extra,
        ])

    def test_classify_end_to_end(self, tmp_path: Path) -> None:
        '''Classify writes assignments with in-plane angles, per-half averages, params and its classifier name.'''
        result = self._invoke(tmp_path, '--no-strict-halfset-independence')
        assert result.exit_code == 0, result.output
        output_dir = tmp_path / 'stamp' / 'classify'
        assignments = json.loads((output_dir / 'class_assignments.json').read_text())
        assert {a['classifier'] for a in assignments} == {'stamp-polish-classifier'}
        assert any(a['inplane_angle_degrees'] is not None for a in assignments)
        posed = [a for a in assignments if a['cluster_id'] != 'noise']
        assert posed and all(a['rotation'] is not None and len(a['shift_voxels']) == 3 and a['cc'] is not None and a['polarity'] is None for a in posed)
        averages = [path.name for path in (output_dir / 'class_averages').glob('*.mrc')]
        assert any('halfA' in name for name in averages) and any('halfB' in name for name in averages)
        assert (output_dir / 'params.toml').exists()

    def test_class_assignment_carries_vesicle_id(self, tmp_path: Path) -> None:
        '''vesicle_id on a Particle survives unchanged into its ClassAssignment, and stays None when unset.'''
        expected: dict[str, str | None] = {}
        def add_vesicles(payload: dict) -> None:
            for index, particle in enumerate(payload['particles']):
                particle['vesicle_id'] = f'tomo000:v{index % 2 + 1:04d}' if index % 3 else None
                expected[particle['particle_id']] = particle['vesicle_id']
        result = self._invoke(tmp_path, '--no-strict-halfset-independence', edit=add_vesicles)
        assert result.exit_code == 0, result.output
        assignments = json.loads((tmp_path / 'stamp' / 'classify' / 'class_assignments.json').read_text())
        assert assignments and all(a['vesicle_id'] == expected[a['particle_id']] for a in assignments)     # particles near the edge are skipped, so only those classified are compared

    def test_polish_strict_halfsets(self, tmp_path: Path) -> None:
        '''Strict mode runs polish per half, matches clusters by reference and writes pose fields for both halves.'''
        result = self._invoke(tmp_path, '--strict-halfset-independence', '--bin-factor', '1', '--null-sigmas', '0.5')
        assert result.exit_code == 0, result.output
        assert 'Cross-half cluster matching' in result.output
        output_dir = tmp_path / 'stamp' / 'classify'
        assignments = json.loads((output_dir / 'class_assignments.json').read_text())
        assert {a['classifier'] for a in assignments} == {'stamp-polish-classifier-strict'}
        assert {a['cluster_id'] for a in assignments if a['cluster_id'] != 'noise'}
        halves = {path.name.split('_half')[1][0] for path in (output_dir / 'class_averages').glob('*.mrc')}
        assert halves == {'A', 'B'}

    def test_polish_runs_for_every_prior(self, tmp_path: Path) -> None:
        '''The axis and none priors run end to end from the same particle set, `none` without any orientation.'''
        for prior in ('axis', 'none'):
            root = tmp_path / prior
            root.mkdir()
            particle_set_path, raw_dir = _build_dataset(root)
            payload = json.loads(particle_set_path.read_text())
            payload['orientation_prior'] = prior
            if prior == 'none':
                for particle in payload['particles']:
                    del particle['orientation']
            particle_set_path.write_text(json.dumps(payload))
            result = runner.invoke(stamp_app, [
                '-d', str(root), 'classify', '--particles', str(particle_set_path), '--raw-dir', str(raw_dir), '--out-dir', str(root),
                '--voxel-size-a', '10.0', '--box-length-a', '200.0', '--method', 'leiden', '--min-cluster-size', '10', '--n-components', '10',
                '--tilt-max', '60', '--bin-factor', '1', '--null-sigmas', '0.5', '--n-rots', '150', '--no-strict-halfset-independence',
            ])
            assert result.exit_code == 0, result.output
            assignments = json.loads((root / 'stamp' / 'classify' / 'class_assignments.json').read_text())
            posed = [a for a in assignments if a['cluster_id'] != 'noise']
            assert posed and all((a['polarity'] in (1, -1)) == (prior == 'axis') for a in posed)

# TestBuildClassifyCommands: class containing tests for the classify argv built from the run config
class TestBuildClassifyCommands:
    # test_build_classify_commands_passes_polish_settings: the wedge and alignment settings reach the classify argv
    def test_build_classify_commands_passes_polish_settings(self, tmp_path) -> None:
        config = RunConfig.model_validate({**_CONFIG_DICT, 'run': {**_CONFIG_DICT['run'], 'tilt_max_degrees': 60.0, 'tilt_axis_degrees': 85.0}, 'stage': {**_CONFIG_DICT['stage'], 'classify': {'method': 'kmeans', 'n_rotations': 500}}})
        argv = build_classify_commands(config, tmp_path)[0].argv
        for flag, value in (('--method', 'kmeans'), ('--n-rots', '500'), ('--tilt-max', '60.0'), ('--tilt-axis', '85.0')):
            assert argv[argv.index(flag) + 1] == value
        default = build_classify_commands(RunConfig.model_validate(_CONFIG_DICT), tmp_path)[0].argv
        assert '--tilt-max' not in default and default[default.index('--method') + 1] == 'leiden'
