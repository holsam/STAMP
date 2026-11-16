'''
STAMP: integration tests for `stamp refine prepare` and `stamp refine assess`
'''

# Import external dependencies
import json, mrcfile, numpy as np, pytest, starfile
from typer.testing import CliRunner
from stamp.cli.cli import stamp_app
from stamp.utils.errors import StampValidationError

# Initialise runner
runner = CliRunner()

# _fixture: a two-half class with per-half Stage D averages, plus the three input JSON files
def _fixture(tmp_path, mixed=False):
    stage_d = tmp_path / 'stage_d'
    averages = stage_d / 'class_averages'
    averages.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for half in ('A', 'B'):
        with mrcfile.new(averages / f'c00_half{half}_n20.mrc', overwrite=True) as mrc:
            mrc.set_data(rng.random((16, 16, 16)).astype(np.float32))
            mrc.voxel_size = 3.0

    halves = ['A', 'A'] if mixed else ['A', 'B']
    particles = {
        'particles': [
            {
                'particle_id': f'p{i}',
                'tomogram_id': 't0',
                'position': [8.0, 8.0, 8.0],
                'source_picker': 'x',
                'half_set': halves[i],
                'prior': 'normal',
            }
            for i in range(2)
        ],
        'consensus_rule': 'intersection', 
        'contributing_pickers': ['x'],
    }
    (tmp_path / 'particle_set.json').write_text(json.dumps(particles))
    (stage_d / 'class_assignments.json').write_text(json.dumps(
        [
            {
                'particle_id': f'p{i}',
                'cluster_id': 'c00',
                'classifier': 'k'
            } 
        for i in range(2)]
    ))
    (tmp_path / 'identification.json').write_text(json.dumps(
        [
            {
                'cluster_id': 'c00',
                'candidate_protein': 'x',
                'fit_score': 0.9,
                'method': 'm',
                'score_gap_to_runner_up': 0.3
            }
        ]
    ))
    (tmp_path / 'tomo').mkdir()
    return stage_d

def _prepare(tmp_path, stage_d):
    return runner.invoke(stamp_app, [
        '-d', str(tmp_path),
        'refine', 'prepare',
        '--class-id', 'c00',
        '--identification', str(tmp_path / 'identification.json'),
        '--particles', str(tmp_path / 'particle_set.json'),
        '--class-assignments', str(stage_d / 'class_assignments.json'),
        '--raw-dir', str(tmp_path / 'tomo'),
        '--out-dir', str(tmp_path),
        '--voxel-size-a', '3.0',
    ])

def _assess(tmp_path, half_a, half_b, *extra):
    return runner.invoke(stamp_app, [
        '-d', str(tmp_path),
        'refine', 'assess',
        '--class-id', 'c00',
        '--prepared-dir', str(tmp_path / 'stamp' / 'refine' / 'c00'),
        '--half-a', str(half_a),
        '--half-b', str(half_b),
        *extra,
    ])

def _write_refined(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = 3.0

# TestRefinePrepare: class containing tests for `stamp refine prepare`
class TestRefinePrepare:
    def test_prepare_writes_disjoint_halves(self, tmp_path):
        stage_d = _fixture(tmp_path)
        result = _prepare(tmp_path, stage_d)
        assert result.exit_code == 0, result.output
        combined = tmp_path / 'stamp' / 'refine' / 'c00'
        for half in ('halfA', 'halfB'):
            assert (combined / half / 'particles.star').is_file()
            assert (combined / half / 'reference.mrc').is_file()
            assert (combined / half / 'README.md').is_file()
        assert json.loads((combined / 'prepare.json').read_text())['voxel_size_angstrom'] == 3.0
        rows_a = starfile.read(combined / 'halfA' / 'particles.star')
        rows_b = starfile.read(combined / 'halfB' / 'particles.star')
        assert list(rows_a['rlnTomoParticleName']) == ['p0'] and list(rows_b['rlnTomoParticleName']) == ['p1']

    def test_prepare_rejects_half_set_mixed_class(self, tmp_path):
        stage_d = _fixture(tmp_path, mixed=True)
        result = _prepare(tmp_path, stage_d)
        assert result.exit_code != 0
        assert not (tmp_path / 'stamp' / 'refine' / 'c00' / 'halfA' / 'particles.star').exists()

# TestRefineAssess: class containing tests for `stamp refine assess`
class TestRefineAssess:
    def _prepared(self, tmp_path):
        stage_d = _fixture(tmp_path)
        assert _prepare(tmp_path, stage_d).exit_code == 0
        return stage_d, tmp_path / 'stamp' / 'refine' / 'c00'

    def test_assess_end_to_end(self, tmp_path):
        _, prepared = self._prepared(tmp_path)
        rng = np.random.default_rng(1)
        signal = np.asarray(mrcfile.open(prepared / 'halfA' / 'reference.mrc').data, dtype=np.float32)
        _write_refined(tmp_path / 'runA' / 'final.mrc', signal + 0.1 * rng.random(signal.shape))
        _write_refined(tmp_path / 'runB' / 'final.mrc', signal + 0.1 * rng.random(signal.shape))
        result = _assess(tmp_path, tmp_path / 'runA' / 'final.mrc', tmp_path / 'runB' / 'final.mrc')
        assert result.exit_code == 0, result.output
        assert (prepared / 'fsc.txt').is_file() and (prepared / 'fsc.svg').is_file()
        assert (prepared / 'params.toml').is_file()

    def test_assess_rejects_same_directory(self, tmp_path):
        self._prepared(tmp_path)
        rng = np.random.default_rng(2)
        _write_refined(tmp_path / 'runA' / 'final.mrc', rng.random((16, 16, 16)))
        _write_refined(tmp_path / 'runA' / 'final_b.mrc', rng.random((16, 16, 16)))
        result = _assess(tmp_path, tmp_path / 'runA' / 'final.mrc', tmp_path / 'runA' / 'final_b.mrc')
        assert result.exit_code != 0
        assert isinstance(result.exception, StampValidationError) and 'share a directory' in str(result.exception)

    def test_assess_rejects_unrefined_seed(self, tmp_path):
        _, prepared = self._prepared(tmp_path)
        rng = np.random.default_rng(3)
        seed = np.asarray(mrcfile.open(prepared / 'halfA' / 'reference.mrc').data, dtype=np.float32)
        _write_refined(tmp_path / 'runA' / 'final.mrc', seed)
        _write_refined(tmp_path / 'runB' / 'final.mrc', rng.random(seed.shape))
        result = _assess(tmp_path, tmp_path / 'runA' / 'final.mrc', tmp_path / 'runB' / 'final.mrc')
        assert result.exit_code != 0
        assert isinstance(result.exception, StampValidationError) and 'identical to its seed' in str(result.exception)
