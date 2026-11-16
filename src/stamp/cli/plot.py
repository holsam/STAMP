'''
STAMP: standalone plot generation CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import plots command functions/variables
from stamp.utils.options.cli import NProcessesOption, OptionalRawDirOption, OutDirOption, PickPlot3dOption, PickPlotStyleOption, PickZstackMovieOption, PlotFormatOption, SegDirOption
import stamp.commands.plot as plotFuncs
from stamp.utils.options.literals import PlotFormat

# Initialise Typer app
plotCli = typer.Typer(no_args_is_help = True, add_completion = False)

# Define pick subcommand
@plotCli.command('pick')
def pick(
    particle_set: Annotated[
        Path,
        typer.Argument(help='particle_set.json (from stamp pick) or decoy_particle_set.json (from stamp decoy).', exists=True, dir_okay=False),
    ],
    segmentation_dir: SegDirOption = None,
    raw_tomogram_dir: OptionalRawDirOption = None,
    output_dir: OutDirOption = Path('.'),
    pick_plot_style: PickPlotStyleOption = 'segmented',
    plot_format: PlotFormatOption = 'tiff',
    pick_zstack_movie: PickZstackMovieOption = True,
    pick_plot_3d: PickPlot3dOption = False,
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
    n_workers: NProcessesOption = 1,
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
