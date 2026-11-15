'''
STAMP: consensus picking CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import pick command functions/variables
from stamp.utils.options.cli import KeepRawOption, NProcessesOption, OutDirOption, PickPlot3dOption, PickPlotStyleOption, PickZstackMovieOption, PlotFormatOption, PlotsOption, RawDirOption, SegDirOption, VoxelSizeOption
import stamp.commands.pick as pickfuncs
from stamp.schemas.particles import OrientationPrior
from stamp.utils.io import resolve_output_dir
from stamp.utils.log import log_started
from stamp.utils.options.literals import PlotFormat

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
    raw_tomogram_dir: RawDirOption,
    segmentation_dir: SegDirOption = None,
    output_dir: OutDirOption = Path('.'),
    voxel_size_angstrom: VoxelSizeOption = None,
    orientation_prior: Annotated[
        OrientationPrior,
        typer.Option('--orientation-prior', help='Default orientation prior for picks that do not state one in a prior column (CSV) or key (JSON): normal (membrane proteins, orientation inferred from --seg-dir), axis (filaments, every pick needs an orientation) or none (soluble particles, orientations are ignored).'),
    ] = OrientationPrior.NORMAL,
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
    n_workers: NProcessesOption = 1,
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
    make_plots: PlotsOption = False,
    pick_plot_style: PickPlotStyleOption = 'segmented',
    plot_format: PlotFormatOption = 'tiff',
    pick_zstack_movie: PickZstackMovieOption = True,
    pick_plot_3d: PickPlot3dOption = False,
    keep_raw: KeepRawOption = False,
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
        orientation_prior=orientation_prior,
    )
