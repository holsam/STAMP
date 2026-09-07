'''
STAMP: unit tests for backends
'''

# Import external dependencies
import json, subprocess, pytest
from pathlib import Path

# Import backends
from stamp.backends import cluster as cluster_module
from stamp.backends.base import ToolCommand
from stamp.backends.cluster import ClusterRunner
from stamp.backends.factory import select_runner
from stamp.backends.local import LocalRunner
from stamp.backends.mock import MockRunner
from stamp.backends.slurm import job_state, render_job_script, submit
from stamp.schemas.cluster_profile import ClusterProfile

# _echo_command: shared function that echos hello-stamp
def _echo_command(tmp_path: Path) -> ToolCommand:
    return ToolCommand(
        tool='echo-test',
        argv=['echo', 'hello-stamp'],
        working_directory=tmp_path,
        output_paths=[],
    )

def _command(tmp_path):
    return ToolCommand(
        tool='relion_refine',
        argv=['relion_refine', '--i', 'x.star'],
        working_directory=tmp_path, 
        output_paths=[]
    )

# TestLocalBackend: class containing unit tests for local backend
class TestLocalBackend:
    def test_local_runner_executes_real_subprocess(self, tmp_path: Path) -> None:
        result = LocalRunner().run(_echo_command(tmp_path))
        assert result.succeeded
        assert 'hello-stamp' in result.stdout

    def test_local_runner_creates_working_directory(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / 'not-yet-created'
        command = ToolCommand(
            tool='echo-test',
            argv=['echo', 'hello'],
            working_directory=missing_dir,
            output_paths=[],
        )
        LocalRunner().run(command)
        assert missing_dir.exists()

# TestMockBackend: class containing unit tests for local backend
class TestMockBackend:
    def test_mock_runner_never_executes(self, tmp_path: Path) -> None:
        result = MockRunner().run(_echo_command(tmp_path))
        assert result.succeeded
        assert 'echo-test' in result.stdout

    def test_mock_runner_can_report_failure(self, tmp_path: Path) -> None:
        result = MockRunner(exit_code=1).run(_echo_command(tmp_path))
        assert not result.succeeded

# TestBackends: class containing unit tests for multiple backends
class TestBackends:
    def test_local_and_mock_results_have_matching_shape(self, tmp_path: Path) -> None:
        command = _echo_command(tmp_path)
        local_result = LocalRunner().run(command)
        mock_result = MockRunner().run(command)
        assert local_result.tool == mock_result.tool
        assert local_result.output_paths == mock_result.output_paths
        assert isinstance(local_result.exit_code, int) and isinstance(mock_result.exit_code, int)
    
# TestSlurmBackend: class containing unit tests for SLURM backend helpers
    def test_script_adds_gpus_only_when_required(self, tmp_path):
        profile = ClusterProfile(partition='cpu', gpus=2, cpus_per_task=4, module_loads=['relion/5.0'])
        gpu_script = render_job_script(_command(tmp_path), True, profile, tmp_path)
        cpu_script = render_job_script(_command(tmp_path), False, profile, tmp_path)
        assert '--partition=cpu' in gpu_script and '--partition=cpu' in cpu_script
        assert '--gpus=2' in gpu_script and '--gpus' not in cpu_script
        assert '--cpus-per-task=4' in gpu_script
        assert 'module load relion/5.0' in gpu_script

    def test_submit_adds_dependency_when_given(self, monkeypatch, tmp_path):
        seen = {}
        def _fake(argv, **kwargs):
            seen['argv'] = argv
            return subprocess.CompletedProcess(argv, 0, stdout='123456;cluster\n', stderr='')
        monkeypatch.setattr(subprocess, 'run', _fake)
        assert submit(tmp_path / 's.sbatch', dependency_ids=['1', '2']) == '123456'
        assert '--dependency=afterok:1:2' in seen['argv']

    def test_submit_omits_dependency_flag_when_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(subprocess, 'run', lambda argv, **k: subprocess.CompletedProcess(argv, 0, stdout='1;c\n', stderr=''))
        submit(tmp_path / 's.sbatch')

    @pytest.mark.parametrize('sacct_out, expected', [('COMPLETED\nCOMPLETED\n', 'COMPLETED'), ('RUNNING\n', 'RUNNING'), ('CANCELLED by 1001\n', 'CANCELLED'), ('', 'PENDING')])
    def test_job_state_parsing(self, monkeypatch, sacct_out, expected):
        monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=sacct_out, stderr=''))
        assert job_state('123456') == expected

# TestClusterBackend: class containing unit tests for cluster backend
class TestClusterBackend:
    def test_first_call_submits_and_returns_immediately(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cluster_module, 'submit', lambda _path: '777')
        result = ClusterRunner(ClusterProfile(partition='a')).run(_command(tmp_path))
        assert result.succeeded  # submission succeeded, not "job finished"
        assert json.loads((tmp_path / '.stamp_cluster_submitted.json').read_text())['job_id'] == '777'

    def test_second_call_while_running_does_not_resubmit(self, monkeypatch, tmp_path):
        (tmp_path / '.stamp_cluster_submitted.json').write_text(json.dumps({'job_id': '42', 'tool': 'x'}))
        monkeypatch.setattr(cluster_module, 'job_state', lambda _id: 'RUNNING')
        monkeypatch.setattr(cluster_module, 'submit', lambda _p: pytest.fail('should not resubmit'))
        result = ClusterRunner(ClusterProfile(partition='a')).run(_command(tmp_path))
        assert (tmp_path / '.stamp_cluster_submitted.json').exists()  # marker kept, still pending

    def test_second_call_after_completion_reports_result_and_clears_marker(self, monkeypatch, tmp_path):
        (tmp_path / '.stamp_cluster_submitted.json').write_text(json.dumps({'job_id': '999', 'tool': 'x'}))
        (tmp_path / 'slurm-999.out').write_text('done')
        monkeypatch.setattr(cluster_module, 'job_state', lambda _id: 'COMPLETED')
        result = ClusterRunner(ClusterProfile(partition='a')).run(_command(tmp_path))
        assert result.succeeded and result.stdout == 'done'
        assert not (tmp_path / '.stamp_cluster_submitted.json').exists()

    def test_failed_job_is_non_zero(self, monkeypatch, tmp_path):
        (tmp_path / '.stamp_cluster_submitted.json').write_text(json.dumps({'job_id': '999', 'tool': 'x'}))
        (tmp_path / 'slurm-999.err').write_text('oom')
        monkeypatch.setattr(cluster_module, 'job_state', lambda _id: 'FAILED')
        result = ClusterRunner(ClusterProfile(partition='a')).run(_command(tmp_path))
        assert not result.succeeded and 'oom' in result.stderr

    def test_missing_sbatch_raises(self, monkeypatch):
        monkeypatch.setattr('shutil.which', lambda _name: None)
        with pytest.raises(SystemExit, match='sbatch'):
            select_runner('cluster')
