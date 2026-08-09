'''
STAMP: abstractions for runners
'''

# Import external dependencies
from pathlib import Path
from pydantic import BaseModel
from typing import Protocol

# ToolCommand: class for a fully-built command and metadata
class ToolCommand(BaseModel):
    tool: str
    argv: list[str]
    working_directory: Path
    output_paths: list[Path] = []

# RunResult: class for result of executing a ToolCommand
class RunResult(BaseModel):
    tool: str
    exit_code: int
    stdout: str
    stderr: str
    output_paths: list[Path]
    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

# Runner: abstraction for any given runner
class Runner(Protocol):
    def run(self, command: ToolCommand) -> RunResult: ...
