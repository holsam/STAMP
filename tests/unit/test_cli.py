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
COMMANDS = ['pick', 'decoy', 'classify', 'identify', 'refine', 'run']

# TestCli: class containing CLI unit tests
class TestCli:
    def test_help_lists_all_commands(self):
        '''All commands should appear in help text'''
        result = runner.invoke(stamp, ['--help'])
        assert result.exit_code == 0
        for command in COMMANDS:
            assert command in result.output
