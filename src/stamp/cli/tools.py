'''
STAMP: CLI for `stamp tools` command
'''

# Import external dependencies
from pathlib import Path
from typing import Annotated
import typer

# Import modules for tools commands
from stamp.tools.diagram import diagram

# Initialise Typer class
toolsCli = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
)

# Import diagram command & subcommands
toolsCli.add_typer(
    diagram.diagramCli,
    name='diagram',
    help='Generate diagrams of STAMP workflows.',
)
