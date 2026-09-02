'''
STAMP: unit tests for CLI
'''

# Import external dependencies
import pytest
from typer.testing import CliRunner

# Import main CLI
from stamp.cli.cli import stamp

# Initialise runner
runner = CliRunner()

# Define constants
STUB_COMMANDS = ['identify', 'refine', 'pipeline']

# TestCli: class containing CLI unit tests
class TestCli:
    def test_help_lists_all_commands(self):
        '''All commands should appear in help text'''
        result = runner.invoke(stamp, ['--help'])
        assert result.exit_code == 0
        for command in STUB_COMMANDS:
            assert command in result.output
    
    @pytest.mark.parametrize('command', STUB_COMMANDS)
    def test_stub_command_reports_not_implemented(self, command: str):
        '''All stubbed commands should report not yet implemented'''
        result = runner.invoke(stamp, [command])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output
