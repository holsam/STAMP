'''
STAMP: unit tests for adapters
'''

# Import external dependencies
import pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.adapters.base import AdapterInputs
from stamp.adapters.mock import MOCK_ADAPTERS, get_mock_adapter
from stamp.backends.mock import MockRunner

# TestAdapters: class containing adapter unit tests
class TestAdapters:
    @pytest.mark.parametrize('name', sorted(MOCK_ADAPTERS))
    def test_mock_adapter_round_trips(self, tmp_path: Path, name: str) -> None:
        adapter = get_mock_adapter(name)
        inputs = AdapterInputs(input_paths=[], output_directory=tmp_path)

        command = adapter.build_command(inputs)
        result = MockRunner().run(command)
        output = adapter.parse_output(result)

        assert command.tool == name
        assert result.succeeded
        assert output.output_paths == []

    def test_unknown_adapter_name_raises_clear_error(self) -> None:
        with pytest.raises(KeyError, match='not-a-real-tool'):
            get_mock_adapter('not-a-real-tool')

    def test_capability_metadata(self) -> None:
        # Spot-check capability metadata
        assert MOCK_ADAPTERS['membrain-pick'].mac_compatible is True
        assert MOCK_ADAPTERS['tomotwin'].mac_compatible is True
        assert MOCK_ADAPTERS['domainfit'].mac_compatible is True
        assert MOCK_ADAPTERS['relion'].mac_compatible is False
        assert MOCK_ADAPTERS['relion'].requires_gpu is True
        assert MOCK_ADAPTERS['stamp-native'].mac_compatible is True

    def test_all_nine_tools_present(self) -> None:
        expected = {
            'stamp-native',
            'membrain-pick',
            'pytom-match-pick',
            'tomotwin',
            'disca',
            'relion',
            'm-refine',
            'domainfit',
        }
        assert set(MOCK_ADAPTERS) == expected