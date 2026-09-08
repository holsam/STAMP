'''
STAMP: CLI for `stamp tools diagram` command
'''

# Import external dependencies
from pathlib import Path
from typing import Annotated
import typer

# Initialise Typer class
diagramCli = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
)

# overview_subcommand: define subcommand for overview diagram
@diagramCli.command('overview', rich_help_panel='Overview Diagrams')
def overview_subcommand():
    '''Generate a diagram of the overall STAMP workflow.'''
    pass

# pick_subcommand: define subcommand for pick diagram
@diagramCli.command('pick', rich_help_panel='Command Diagrams')
def pick_subcommand():
    '''Generate a diagram of the pick command workflow.'''
    pass

# classify_subcommand: define subcommand for classify diagram
@diagramCli.command('classify', rich_help_panel='Command Diagrams')
def classify_subcommand():
    '''Generate a diagram of the classify command workflow.'''
    pass

# identify_subcommand: define subcommand for identify diagram
@diagramCli.command('identify', rich_help_panel='Command Diagrams')
def identify_subcommand():
    '''Generate a diagram of the identify command workflow.'''
    pass
