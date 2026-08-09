'''
STAMP: runner which executes a ToolCommand as a real subprocess
'''

# Import external dependencies
import subprocess

# Import base backend
from stamp.backends.base import RunResult, ToolCommand

# LocalRunner: executes given command directly via subprocess
class LocalRunner:
    def run(self, command: ToolCommand) -> RunResult:
        command.working_directory.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            command.argv,
            cwd=command.working_directory,
            capture_output=True,
            text=True,
        )
        return RunResult(
            tool=command.tool,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            output_paths=command.output_paths,
        )