'''
STAMP: identification CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import identify command functions
import stamp.commands.identify as identifyfuncs

# Initialise Typer app
identifyCli = typer.Typer(no_args_is_help=True, add_completion=False)

# Define identify command
@identifyCli.command()
def identify(
    classes: Annotated[
        Path,
        typer.Option('--classes', exists=True, file_okay=False, help='Directory of class-average MRCs.')
    ],
    candidates: Annotated[
        Path,
        typer.Option('--candidates', exists=True, dir_okay=False, help='Mass-spec candidate YAML.')
    ],
    output_dir: Annotated[
        Path,
        typer.Option('-o', '--output-dir', file_okay=False, help='Output directory.')
    ],
    resolution: Annotated[
        float,
        typer.Option('-r', '--resolution', help='Class-average resolution in Å (used for low-pass filtering).')
    ],
    decoy_classes: Annotated[
        Path | None,
        typer.Option('--decoy-classes', exists=True, file_okay=False, help='Decoy class averages.')
    ] = None,
    backend: Annotated[
        Literal['local', 'mock'],
        typer.Option('--backend', help='Backend to use for processing.')
    ] = 'local',
    fitter: Annotated[
        Literal['native'],
        typer.Option('--fitter', help='Fitter to use for identification.')
    ] = 'native',
    fetch_missing: Annotated[bool, typer.Option('--fetch-missing',
        help='Fetch an AlphaFold model when a candidate has no local structure_path.')] = False,
) -> None:
    '''Fit predicted structures to class averages and score candidates.'''
    identifyfuncs.run_identify(
        classes=classes,
        candidates=candidates,
        output_dir=output_dir,
        decoy_classes=decoy_classes,
        resolution=resolution,
        backend=backend,
        fitter=fitter,
        fetch_missing=fetch_missing,
    )
