'''
STAMP: CLI for `stamp tools filter-picks`
'''

# Import external dependencies
from pathlib import Path
from typing import Annotated
import typer

# Import internal STAMP objects
from stamp.filter.state import load_filter_state

# Initialise Typer app
filterCli = typer.Typer(no_args_is_help=True, add_completion=False)

# filter_picks: launch the interactive filtering GUI
@filterCli.command('filter', rich_help_panel='Utilities')
def filter_picks(
    particle_set: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help='particle_set.json from stamp pick.'),
    ],
    segmentation_dir: Annotated[
        Path | None,
        typer.Option('-s', '--seg-dir', help='Directory of segmentation MRCs, matched to particles by tomogram_id.', exists=True, file_okay=False),
    ] = None,
    raw_tomogram_dir: Annotated[
        Path | None,
        typer.Option('-r', '--raw-dir', help='Directory of raw tomogram MRCs, matched to particles by tomogram_id.', exists=True, file_okay=False),
    ] = None,
) -> None:
    '''Open an interactive GUI to review consensus picks against their segmentation/raw tomogram.'''
    if segmentation_dir is None and raw_tomogram_dir is None:
        raise typer.BadParameter('At least one of --seg-dir or --raw-dir is required.')
    # If tkinter absent (eg headless install), keep rest of CLI importable without it
    try:
        from stamp.filter.app import run_filter_gui
    except ImportError as exc:
        raise typer.BadParameter('tkinter not available, stamp filter command unsupported on headless installs.') from exc
    output_dir = particle_set.parent
    state = load_filter_state(particle_set, segmentation_dir, raw_tomogram_dir, output_dir=output_dir)
    run_filter_gui(state, output_dir)
