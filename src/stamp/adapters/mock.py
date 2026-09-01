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
    automatable: bool = False
    batches_natively: bool = False
    runs_in_process: bool = False
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


# _canned_picks_for: stub parsed-output for a mock picker
def _canned_picks_for(_name: str) -> dict:
    return {'picks': []}


# MOCK_ADAPTERS: instantiate a MockAdapter class for each external tool
MOCK_ADAPTERS: dict[str, MockAdapter] = {
    'stamp-native': MockAdapter(
        'stamp-native',
        stage='pick',
        mac_compatible=True,
        requires_gpu=False,
        automatable=True,
        batches_natively=True,
        runs_in_process=False,
        canned_parsed=_canned_picks_for('stamp-native'),
    ),
    'membrain-pick': MockAdapter(
        'membrain-pick',
        stage='pick',
        mac_compatible=True,
        requires_gpu=False,
        automatable=True,
        batches_natively=False,
        runs_in_process=False,
        canned_parsed=_canned_picks_for('membrain-pick'),
    ),
    'pytom-match-pick': MockAdapter(
        'pytom-match-pick',
        stage='pick',
        mac_compatible=False,
        requires_gpu=True,
        automatable=True,
        batches_natively=False,
        runs_in_process=False,
        canned_parsed=_canned_picks_for('pytom-match-pick'),
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
