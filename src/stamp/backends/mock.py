'''
STAMP: runner which does not execute anything (for tests)
'''

# Import base backend
from stamp.backends.base import RunResult, ToolCommand

# MockRunner: stand-in for runners to use in tests (always report success unless told otherwise)
class MockRunner:
    def __init__(self, fixed_stdout: str = '', exit_code: int = 0) -> None:
        self._fixed_stdout = fixed_stdout
        self._exit_code = exit_code

    def run(self, command: ToolCommand) -> RunResult:
        return RunResult(
            tool=command.tool,
            exit_code=self._exit_code,
            stdout=self._fixed_stdout or f'[mock] ran {command.tool}',
            stderr='',
            output_paths=command.output_paths,
        )
