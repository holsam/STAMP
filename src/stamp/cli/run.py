'''
STAMP: full pipeline CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import run command functions
import stamp.commands.run as runfuncs

# Initialise Typer app
runCli = typer.Typer(no_args_is_help=True, add_completion=False)

# Define run command
@runCli.command()
def run(
    config: Annotated[
        Path,
        typer.Option('--config', exists=True, dir_okay=False, help='stamp_run.toml'),
    ],
    force: Annotated[
        bool,
        typer.Option('--force', help='Re-run every stage.'),
    ] = False,
    from_stage: Annotated[
        str | None,
        typer.Option('--from', help='Re-run from this stage onward.'),
    ] = None,
    no_decoy: Annotated[
        bool,
        typer.Option('--no-decoy', help='Skip the decoy control track.'),
    ] = False,
    backend: Annotated[
        Literal['local', 'mock'] | None,
        typer.Option('--backend', help='Override [run].backend.'),
    ] = None,
) -> None:
    '''Run the STAMP pipeline from a config file.'''
    runfuncs.run_full_pipeline(
        config_path=config,
        force=force,
        from_stage=from_stage,
        no_decoy=no_decoy,
        backend=backend,
    )
