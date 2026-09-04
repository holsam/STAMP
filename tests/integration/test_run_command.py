'''
STAMP: integration tests for `stamp run` command
'''

# Import external dependencies
import json, tomllib
from typer.testing import CliRunner
from stamp.cli.cli import stamp

# Initialise runner
runner = CliRunner()

# TestRunCommand: class containing tests for run command
class TestRunCommand:
    def test_run_produces_both_tracks_and_report(self, tmp_path, run_mock_pipeline):
        out = run_mock_pipeline(tmp_path)
        assert (out / 'real' / 'stage_pick' / 'particle_set.json').is_file()
        assert (out / 'decoy' / 'stage_pick' / 'decoy_particle_set.json').is_file()
        assert (out / 'real' / 'stage_classify').is_dir()

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
        first = runner.invoke(stamp, ['run', '--config', str(config)])
        assert first.exit_code == 0
        state = json.loads((tmp_path / 'out' / 'run_state.json').read_text())
        assert state['real']['pick'] is True

        second = runner.invoke(stamp, ['run', '--config', str(config)])
        assert second.exit_code == 0
        assert 'pick' not in second.output.lower() or 'skip' in second.output.lower()

    def test_no_decoy_runs_real_only(self, tmp_path, run_mock_pipeline):
        out = run_mock_pipeline(tmp_path, args=['--no-decoy'])
        assert not (out / 'decoy').exists()
        assert 'NOT RUN' in (out / 'report.md').read_text()
