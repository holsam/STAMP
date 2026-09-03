'''
STAMP: integration tests for `stamp refine`
'''

# Import external dependencies
import json, mrcfile, numpy as np, pytest
from typer.testing import CliRunner
from stamp.cli.cli import stamp

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
                'half_set': halves[i]
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

def _invoke(tmp_path, stage_d, out):
    return runner.invoke(stamp, [
        'refine',
        '--class-id', 'c00',
        '--identification', str(tmp_path / 'identification.json'),
        '--particles', str(tmp_path / 'particle_set.json'),
        '--class-assignments', str(stage_d / 'class_assignments.json'),
        '--raw-dir', str(tmp_path / 'tomo'),
        '--out-dir', str(out),
        '--voxel-size-a', '3.0',
        '--backend', 'mock',
    ])

# TestRefineCommand: class containing tests for `stamp refine`
class TestRefineCommand:
    def test_refine_mock_writes_independent_halves(self, tmp_path):
        stage_d = _fixture(tmp_path)
        out = tmp_path / 'refine'
        result = _invoke(tmp_path, stage_d, out)
        assert result.exit_code == 0, result.output
        combined = out / 'c00'
        assert (combined / 'final_A.mrc').is_file() and (combined / 'final_B.mrc').is_file()
        assert (combined / 'fsc.txt').is_file() and (combined / 'fsc.svg').is_file()
        assert (combined / 'params.toml').is_file()
        assert (out / 'c00' / 'halfA').is_dir() and (out / 'c00' / 'halfB').is_dir()

    def test_refine_rejects_half_set_mixed_class(self, tmp_path):
        stage_d = _fixture(tmp_path, mixed=True)
        out = tmp_path / 'refine'
        result = _invoke(tmp_path, stage_d, out)
        assert result.exit_code != 0
        assert not (out / 'c00' / 'halfA' / 'particles.star').exists()
