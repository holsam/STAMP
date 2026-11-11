'''
STAMP: consensus picking CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import pick command functions/variables
import stamp.commands.pick as pickfuncs
from stamp.utils.io import resolve_output_dir
from stamp.utils.log import log_started

# Initialise Typer app
pickCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define pick command
@pickCli.command()
def pick(
    picks_files: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, readable=True, help='Picks files to import (.csv, .star, .json, .jsonl); one picker per file.'),
    ],
    segmentation_dir: Annotated[
        Path,
        typer.Option('-s', '--seg-dir', exists=True, file_okay=False, help='Directory of segmented MRC files; filename stems are tomogram IDs. Used to infer orientations.'),
    ],
    raw_tomogram_dir: Annotated[
        Path,
        typer.Option('-r', '--raw-dir', exists=True, file_okay=False, help='Directory of raw tomogram MRCs, matched to segmentations by filename stem.')
    ],
    output_dir: Annotated[
        Path,
        typer.Option('-o', '--out-dir', file_okay=False, help='Directory to write outputs to.')
    ] = Path('.'),
    voxel_size_angstrom: Annotated[
        float | None,
        typer.Option('--voxel-size-a', help='Voxel size of the input tomograms, in Angstrom.')
    ] = None,
    names: Annotated[
        list[str] | None,
        typer.Option('--name', help='Picker name for each picks file, in order (repeatable). Defaults to the file stem.', rich_help_panel = 'Consensus'),
    ] = None,
    consensus_rule: Annotated[
        Literal['union', 'intersection'],
        typer.Option('--consensus-rule', help='Rule to use for consensus reconciliation.', rich_help_panel = 'Consensus'),
    ] = 'union',
    distance_threshold: Annotated[
        float,
        typer.Option('--distance-threshold', help='Voxel distance within which picks are merged.', rich_help_panel = 'Consensus'),
    ] = 15.0,
    half_set_seed: Annotated[
        int,
        typer.Option('--half-set-seed', help='Seed for half-set assignment.', rich_help_panel = 'Consensus'),
    ] = 0,
    n_workers: Annotated[
        int,
        typer.Option('-n', '--n-processes', help='Processes to use for reconciliation, the vesicle summary and plots (1 = sequential).', rich_help_panel = 'Backend'),
    ] = 1,
    max_beam_angle_deviation: Annotated[
        float | None,
        typer.Option('--max-beam-angle-deviation', help='Normal deviation from beam-orthogonal plane to drop particles.', rich_help_panel = 'Filtering'),
    ] = None,
    max_surface_distance_angstrom: Annotated[
        float,
        typer.Option('--max-surface-distance-a', help='Picks further than this from any segmentation surface get no inferred orientation (classify skips them).', rich_help_panel = 'Filtering'),
    ] = 300.0,
    vesicle_labels_mrc: Annotated[
        Path | None,
        typer.Option('--vesicle-labels-mrc', help='Path to an EValuator labelled MRC, or a directory to be matched by filename stem.', exists=True, readable=True, rich_help_panel = 'Filtering'),
    ] = None,
    make_plots: Annotated[
        bool,
        typer.Option('--plots', help='Write consensus-pick position plots.', rich_help_panel = 'Plotting'),
    ] = False,
    pick_plot_style: Annotated[
        Literal['segmented', 'none'],
        typer.Option('--pick-plot-style', help='Plot style to use.', rich_help_panel = 'Plotting'),
    ] = 'segmented',
    plot_format: Annotated[
        Literal['png', 'jpg', 'tiff', 'svg'],
        typer.Option('--plot-format', help='Image format for static plots.', rich_help_panel = 'Plotting'),
    ] = 'tiff',
    pick_zstack_movie: Annotated[
        bool,
        typer.Option('--pick-zstack-movie/--no-pick-zstack-movie', help='Create a per-tomogram movie stepping through Z with picks highlighted.', rich_help_panel = 'Plotting'),
    ] = True,
    pick_plot_3d: Annotated[
        bool,
        typer.Option('--pick-plot-3d/--no-pick-plot-3d', help='Create a static 3D pick visualisation per tomogram.', rich_help_panel = 'Plotting'),
    ] = False,
    keep_raw: Annotated[
        bool,
        typer.Option('--keep-raw', help='Keep the per-picker import cache instead of archiving it to raw.tar.gz.', rich_help_panel = 'Backend'),
    ] = False,
) -> None:
    '''Import external picks, infer orientations, reconcile across pickers, and assign half-sets.'''
    log_started('picking')
    pickfuncs.run_pick(
        picks_files=picks_files,
        picker_names=names,
        segmentation_dir=segmentation_dir,
        raw_tomogram_dir=raw_tomogram_dir,
        output_dir=resolve_output_dir(output_dir, 'pick'),
        voxel_size_angstrom=voxel_size_angstrom,
        consensus_rule=consensus_rule,
        distance_threshold=distance_threshold,
        half_set_seed=half_set_seed,
        n_workers=n_workers,
        make_plots=make_plots,
        pick_plot_style=pick_plot_style,
        plot_format=plot_format,
        pick_zstack_movie=pick_zstack_movie,
        pick_plot_3d=pick_plot_3d,
        max_beam_angle_deviation=max_beam_angle_deviation,
        vesicle_labels_mrc=vesicle_labels_mrc,
        max_surface_distance_angstrom=max_surface_distance_angstrom,
        keep_raw=keep_raw,
    )
