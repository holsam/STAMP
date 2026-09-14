'''
STAMP: CLI entrypoint
'''

# Import external dependencies
import sys, typer
from pathlib import Path
from typing import Annotated, Literal

# Import internal commands
from stamp.cli import classify, decoy, identify, pick, refine, run, tools
from stamp.utils.log import configure_logging, log

# Initialise Typer app 
stamp = typer.Typer(
    name = 'stamp',
    help = 'Sub-Tomogram Averaging Membrane Protein pipeline',
    no_args_is_help = True,
    add_completion = False,
    context_settings={'help_option_names': ['-h', '--help']},
)

# Add command Typer classes
stamp.add_typer(pick.pickCli)
stamp.add_typer(decoy.decoyCli)
stamp.add_typer(classify.classifyCli)
stamp.add_typer(identify.identifyCli)
stamp.add_typer(refine.refineCli)
stamp.add_typer(run.runCli)
stamp.add_typer(
    tools.toolsCli,
    name='tools',
    help='Misc STAMP tools.',
    rich_help_panel='Utilities'
)

# logging_callback: provide logging options and configure logging
@stamp.callback()
def logging_callback(
    ctx: typer.Context,
    log_dir: Annotated[
        Path | None,
        typer.Option('-d', '--directory', help='Directory to write log file to.', show_default=None, rich_help_panel='Logging Options')
    ] = None,
    log_mode: Annotated[
        Literal['append', 'overwrite'],
        typer.Option('-m', '--mode', help='Mode for handling existing log files.', rich_help_panel='Logging Options')
    ] = 'append',
    quiet: Annotated[
        int,
        typer.Option('-q', '--quiet', count=True, help='Decrease logging verbosity.', show_default=False, rich_help_panel='Logging Options', metavar='')
    ] = 0,
    verbosity: Annotated[
        int,
        typer.Option('-v', '--verbose', count=True, help='Increase logging verbosity.', show_default=False, rich_help_panel='Logging Options', metavar='')
    ] = 0,
) -> None:
    if ctx.resilient_parsing or ctx.invoked_subcommand is None:
        return
    if {'--help', '-h'} & set(sys.argv[1:]):
        return
    if quiet and verbosity:
        raise typer.BadParameter('-q/--quiet and -v/--verbose are mutually exclusive.')
    verbosity = min(verbosity, 2)
    quiet = min(quiet, 2)
    log_path, log_level = configure_logging(directory=log_dir, mode=log_mode, quiet=quiet, verbosity=verbosity)
    print()
    log.debug(f'Logging configured to stderr and log file (mode: {log_mode}) at level {log_level}')
    log.info(f'Logs will be saved to: {log_path}')
