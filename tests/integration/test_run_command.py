'''
STAMP: integration tests for `stamp run` command
'''

# Import external dependencies
import json, pytest, tomllib
from typer.testing import CliRunner
from stamp.cli.cli import stamp_app

# Initialise runner
runner = CliRunner()

# TestRunCommand: class containing tests for run command
class TestRunCommand:
    def test_run_produces_both_tracks_and_report(self, tmp_path, run_mock_pipeline):
        out = run_mock_pipeline(tmp_path)
        assert (out / 'pick' / 'particle_set.json').is_file()
        assert (out / 'decoy' / 'decoy_particle_set.json').is_file()
        assert (out / 'classify_real').is_dir()

        sidecars = list(out.rglob('params.toml'))
        assert len(sidecars) >= 4
        for path in sidecars:
            assert tomllib.loads(path.read_text())['stamp_commit'] not in ('', 'unknown')

        report = (out / 'report.md').read_text().splitlines()
        assert report[2].startswith('**DECOY CONTROL:')
        payload = json.loads((out / 'report.json').read_text())
        assert 'identifications' in payload and 'provenance' in payload

    def test_second_invocation_skips_completed_stages(self, tmp_path, make_dataset, write_config):
        make_dataset(tmp_path)
        config = write_config(tmp_path)
        first = runner.invoke(stamp_app, ['-d', str(tmp_path), 'run', '--config', str(config)])
        assert first.exit_code == 0
        state = json.loads((tmp_path / 'stamp' / 'run_state.json').read_text())
        assert state['real']['pick'] is True

        second = runner.invoke(stamp_app, ['-d', str(tmp_path), 'run', '--config', str(config)])
        assert second.exit_code == 0
        assert 'pick' not in second.output.lower() or 'skip' in second.output.lower()

    def test_no_decoy_runs_real_only(self, tmp_path, run_mock_pipeline):
        out = run_mock_pipeline(tmp_path, args=['--no-decoy'])
        assert not (out / 'decoy').exists()
        assert 'NOT RUN' in (out / 'report.md').read_text()

    # test_run_pipeline_for_non_membrane_priors: pick, random-volume decoy and the polish step run end to end with no segmentation, for soluble particles (none) and filaments (axis, picks carry an axis quaternion)
    @pytest.mark.parametrize('prior', ['none', 'axis'])
    def test_run_pipeline_for_non_membrane_priors(self, tmp_path: Path, make_dataset, prior: str) -> None:
        make_dataset(tmp_path)
        if prior == 'axis':
            rows = (tmp_path / 'picks.csv').read_text().splitlines()
            (tmp_path / 'picks.csv').write_text('\n'.join([rows[0] + ',qw,qx,qy,qz'] + [row + ',1,0,0,0' for row in rows[1:]]) + '\n')
        (tmp_path / 'stamp_run.toml').write_text(f"""
    [run]
    raw_tomogram_dir = "tomo"
    output_dir = "."
    voxel_size_angstrom = 13.48
    backend = "mock"
    stop_after = "classify"
    orientation_prior = "{prior}"
    tilt_max_degrees = 60.0

    [decoy]
    enabled = true
    method = "random-volume"
    min_distance_from_picks_angstrom = 40.0
    min_border_angstrom = 120.0

    [stage.pick]
    picks_files = ["picks.csv"]

    [stage.classify]
    method = "leiden"
    min_cluster_size = 10
    box_angstrom = 120
    bin_factor = 1
    n_rotations = 100
    null_sigmas = 0.5
    strict_halfset_independence = false

    [stage.identify]
    candidates = "candidates.yaml"
    resolution = 25.0
    """)
        result = CliRunner().invoke(stamp_app, ['-d', str(tmp_path), 'run', '--config', str(tmp_path / 'stamp_run.toml')])
        assert result.exit_code == 0, result.output
        for classify_dir in (tmp_path / 'stamp' / 'classify_real', tmp_path / 'stamp' / 'classify_decoy'):
            assert (classify_dir / 'class_assignments.json').is_file()
        assert json.loads((tmp_path / 'stamp' / 'pick' / 'particle_set.json').read_text())['orientation_prior'] == prior
        decoy_assignments = json.loads((tmp_path / 'stamp' / 'classify_decoy' / 'class_assignments.json').read_text())
        assert {a['cluster_id'] for a in decoy_assignments} == {'noise'}     # random positions share no structure, polish rejects them all
