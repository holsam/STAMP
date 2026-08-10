'''
STAMP: mock adapter for testing
'''

# Import external dependencies
from dataclasses import dataclass, field

# Import internal STAMP classes
from stamp.adapters.base import AdapterInputs, AdapterOutput
from stamp.backends.base import RunResult, ToolCommand

# MockAdapter: a class for stand-in adapters for named tools
@dataclass
class MockAdapter:
    name: str
    stage: str
    mac_compatible: bool
    requires_gpu: bool
    canned_parsed: dict = field(default_factory=dict)

    def build_command(self, inputs: AdapterInputs) -> ToolCommand:
        return ToolCommand(
            tool=self.name,
            argv=['true'],
            working_directory=inputs.output_directory,
            output_paths=[],
        )

    def parse_output(self, result: RunResult) -> AdapterOutput:
        return AdapterOutput(output_paths=[], parsed=dict(self.canned_parsed))


# MOCK_ADAPTERS: instantiate a MockAdapter class for each external tool
MOCK_ADAPTERS: dict[str, MockAdapter] = {
    'pyseg': MockAdapter(
        'pyseg',
        stage='pick',
        mac_compatible=False,
        requires_gpu=False,
    ),
    'membrain-pick': MockAdapter(
        'membrain-pick',
        stage='pick',
        mac_compatible=True,
        requires_gpu=False,
    ),
    'mpicker': MockAdapter(
        'mpicker',
        stage='pick',
        mac_compatible=False,
        requires_gpu=False,
    ),
    'pytom-match-pick': MockAdapter(
        'pytom-match-pick',
        stage='pick',
        mac_compatible=True,
        requires_gpu=True,
    ),
    'tomotwin': MockAdapter(
        'tomotwin',
        stage='classify',
        mac_compatible=True,
        requires_gpu=False,
    ),
    'disca': MockAdapter(
        'disca',
        stage='classify',
        mac_compatible=False,
        requires_gpu=True,
    ),
    'relion': MockAdapter(
        'relion',
        stage='refine',
        mac_compatible=False,
        requires_gpu=True,
    ),
    'm-refine': MockAdapter(
        'm-refine',
        stage='refine',
        mac_compatible=False,
        requires_gpu=True,
    ),
    'domainfit': MockAdapter(
        'domainfit',
        stage='refine',
        mac_compatible=True,
        requires_gpu=False,
    ),
}

# get_mock_adapter: returns a MockAdapter by looking up given name, raising error if unknown
def get_mock_adapter(name: str) -> MockAdapter:
    try:
        return MOCK_ADAPTERS[name]
    except KeyError as exc:
        known = ', '.join(sorted(MOCK_ADAPTERS))
        raise KeyError(f'No mock adapter named {name!r}. Known adapters: {known}') from exc