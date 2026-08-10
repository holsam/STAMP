'''
STAMP: adapter protocol
'''

# Import external dependencies
from pathlib import Path
from pydantic import BaseModel
from typing import Protocol

# Import base backend
from stamp.backends.base import RunResult, ToolCommand

# AdapterInputs: class to hold generic inputs to an adapter
class AdapterInputs(BaseModel):
    input_paths: list[Path] = []
    output_directory: Path
    parameters: dict = {}

# AdapterOutput: class to hold generic parsed output of a completed tool run
class AdapterOutput(BaseModel):
    output_paths: list[Path] = []
    parsed: dict = {}

# ToolAdapter: class that defines an adapter that can build a command and parse the output of a specific tool
class ToolAdapter:
    name: str
    stage: str
    mac_compatible: bool
    requires_gpu: bool

    def build_command(self, inputs: AdapterInputs) -> ToolCommand: ...

    def parse_output(self, result: RunResult) -> AdapterOutput: ...
