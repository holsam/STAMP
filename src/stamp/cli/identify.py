'''
STAMP: identification CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import identify command functions
import stamp.commands.identify as identifyfuncs
from stamp.utils.io import resolve_output_dir

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
    resolution: Annotated[
        float,
        typer.Option('-r', '--resolution', help='Class-average resolution in Å (used for low-pass filtering).')
    ],
    output_dir: Annotated[
        Path,
        typer.Option('-o', '--output-dir', file_okay=False, help='Output directory.')
    ] = Path('.'),
    decoy_classes: Annotated[
        Path | None,
        typer.Option('--decoy-classes', exists=True, file_okay=False, help='Decoy class averages.')
    ] = None,
    backend: Annotated[
        Literal['local', 'mock'],
        typer.Option('--backend', help='Backend to use for processing.', rich_help_panel = 'Backend')
    ] = 'local',
    fitter: Annotated[
        Literal['native'],
        typer.Option('--fitter', help='Fitter to use for identification.', rich_help_panel = 'Backend')
    ] = 'native',
    fetch_missing: Annotated[
        bool,
        typer.Option('--fetch-missing', help='Fetch an AlphaFold model when a candidate has no local structure_path.', rich_help_panel = 'Backend'),
    ] = False,
    make_plots: Annotated[
        bool,
        typer.Option('--plots/--no-plots', help='Write fit-score heatmap and decoy-control histogram plots.', rich_help_panel = 'Plotting'),
    ] = True,
    plot_format: Annotated[
        Literal['png', 'jpg', 'tiff', 'svg'],
        typer.Option('--plot-format', help='Image format for static plots.', rich_help_panel = 'Plotting'),
    ] = 'tiff',
) -> None:
    '''Fit predicted structures to class averages and score candidates.'''
    identifyfuncs.run_identify(
        classes=classes,
        candidates=candidates,
        output_dir=resolve_output_dir(output_dir, 'identify'),
        decoy_classes=decoy_classes,
        resolution=resolution,
        backend=backend,
        fitter=fitter,
        fetch_missing=fetch_missing,
        make_plots=make_plots,
        plot_format=plot_format,
    )
