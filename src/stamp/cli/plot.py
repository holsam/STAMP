'''
STAMP: standalone plot generation CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import plots command functions/variables
import stamp.commands.plot as plotFuncs

# Initialise Typer app
plotCli = typer.Typer(no_args_is_help = True, add_completion = False)

# Define pick subcommand
@plotCli.command('pick')
def pick(
    particle_set: Annotated[
        Path,
        typer.Argument(help='particle_set.json (from stamp pick) or decoy_particle_set.json (from stamp decoy).', exists=True, dir_okay=False),
    ],
    segmentation_dir: Annotated[
        Path | None,
        typer.Option('-s', '--seg-dir', help='Directory of segmented MRC files; filename stems are tomogram IDs.', exists=True, file_okay=False),
    ] = None,
    raw_tomogram_dir: Annotated[
        Path | None,
        typer.Option('-r', '--raw-dir', help='Directory of raw tomogram MRCs, matched to segmentations by filename stem.', exists=True, file_okay=False),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option('-o', '--out-dir', file_okay=False, help='Directory to write plots to.'),
    ] = Path('.'),
    pick_plot_style: Annotated[
        Literal['segmented', 'none'],
        typer.Option('--pick-plot-style', help='Plot style to use.', rich_help_panel='Plotting Options'),
    ] = 'segmented',
    plot_format: Annotated[
        Literal['png', 'jpg', 'tiff', 'svg'],
        typer.Option('--plot-format', help='Image format for static plots.', rich_help_panel='Plotting Options'),
    ] = 'tiff',
    pick_zstack_movie: Annotated[
        bool,
        typer.Option('--pick-zstack-movie/--no-pick-zstack-movie', help='Create a per-tomogram movie stepping through Z with picks highlighted.', rich_help_panel='Plotting Options'),
    ] = True,
    pick_plot_3d: Annotated[
        bool,
        typer.Option('--pick-plot-3d/--no-pick-plot-3d', help='Create a static 3D pick visualisation per tomogram.', rich_help_panel='Plotting Options'),
    ] = False,
    tomogram_ids: Annotated[
        str | None,
        typer.Option('--tomogram-ids', help='Comma-separated tomogram IDs to plot. Mutually exclusive with --n-tomograms.', rich_help_panel = 'Selection'),
    ] = None,
    n_tomograms: Annotated[
        int | None,
        typer.Option('--n-tomograms', help='Randomly select this many tomograms to plot. Mutually exclusive with --tomogram-ids.', rich_help_panel = 'Selection'),
    ] = None,
    seed: Annotated[
        int,
        typer.Option('--seed', help='Seed for random tomogram selection.', rich_help_panel = 'Selection'),
    ] = 0,
    n_workers: Annotated[
        int,
        typer.Option('-n', '--n-processes', help='Processes to use for plot rendering (1 = sequential).'),
    ] = 1,
) -> None:
    '''Render pick/decoy position plots for an existing particle set using a subset of tomograms.'''
    if tomogram_ids is not None and n_tomograms is not None:
        raise typer.BadParameter('--tomogram-ids and --n-tomograms are mutually exclusive.')

    ids = [t.strip() for t in tomogram_ids.split(',') if t.strip()] if tomogram_ids else None

    plotFuncs.run_plots_pick(
        particle_set_path=particle_set,
        output_dir=output_dir,
        segmentation_dir=segmentation_dir,
        raw_tomogram_dir=raw_tomogram_dir,
        pick_plot_style=pick_plot_style,
        plot_format=plot_format,
        pick_zstack_movie=pick_zstack_movie,
        pick_plot_3d=pick_plot_3d,
        tomogram_ids=ids,
        n_tomograms=n_tomograms,
        seed=seed,
        n_workers=n_workers,
    )
