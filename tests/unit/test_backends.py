'''
STAMP: unit tests for backends
'''

# Import external dependencies
import pytest
from pathlib import Path

# Import backends
from stamp.backends.base import ToolCommand
from stamp.backends.local import LocalRunner
from stamp.backends.mock import MockRunner

# _echo_command: shared function that echos hello-stamp
def _echo_command(tmp_path: Path) -> ToolCommand:
    return ToolCommand(
        tool='echo-test',
        argv=['echo', 'hello-stamp'],
        working_directory=tmp_path,
        output_paths=[],
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