'''
STAMP: adapter protocol
'''

# Import external dependencies
from pathlib import Path
from pydantic import BaseModel
from typing import Protocol

# Import base backend
from stamp.backends.base import RunResult, ToolCommand

# AdapterInputs: class to hold generic inputs to an adapter
class AdapterInputs(BaseModel):
    input_paths: list[Path] = []
    raw_tomogram_paths: list[Path] = []
    tomogram_ids: list[str] = []
    output_directory: Path
    parameters: dict = {}

# AdapterOutput: class to hold generic parsed output of a completed tool run
class AdapterOutput(BaseModel):
    output_paths: list[Path] = []
    parsed: dict = {}

# ToolAdapter: class that builds a command for, and parses the output of, one picker
class ToolAdapter(Protocol):
    name: str
    stage: str
    mac_compatible: bool
    requires_gpu: bool
    automatable: bool
    batches_natively: bool
    runs_in_process: bool

    def build_command(self, inputs: AdapterInputs) -> ToolCommand: ...

    def parse_output(self, result: RunResult) -> AdapterOutput: ...
