'''
STAMP: CLI for `stamp config` command
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated

# Import internal stamp objects
import stamp.config.config as configFuncs
from stamp.utils.errors import StampError
from stamp.utils.log import log

# Initialise Typer app
configCli = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
)

@configCli.command('init')
def init_subcommand(
    path: Annotated[
        Path,
        typer.Option(help='Directory to write stamp_run.toml into, or path to stamp_run.toml.')
    ] = Path('.'),
    force: Annotated[
        bool,
        typer.Option('-f', '--force', help='Overwrite an existing stamp_run.toml.')
    ] = False,
) -> None:
    '''Write a blank, annotated stamp_run.toml.'''
    configFuncs.init_config(path, force)

@configCli.command('edit')
def edit_subcommand(
    path: Annotated[
        Path,
        typer.Argument(help='Directory containing stamp_run.toml, or path to stamp_run.toml.')
    ] = Path('.'),
    editor: Annotated[
        str | None,
        typer.Option('--editor', help='Editor to use instead of $EDITOR.', show_default=False)
    ] = None
) -> None:
    '''Edit a stamp_run.toml file.'''
    configFuncs.edit_config(path, editor)

# -- config_show: prints the config file's contents
@configCli.command('show')
def config_show(
    path: Annotated[
        Path,
        typer.Argument(help='Directory containing stamp_run.toml, or path to stamp_run.toml.')
    ] = Path('.'),
):
    '''
    Print a stamp_run.toml config file, or a message if none exists.
    '''
    configFuncs.show_config(path)
