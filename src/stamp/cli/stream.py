'''
STAMP: stream command group CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import stream command functions
import stamp.commands.stream as streamfuncs
from stamp.schemas.particles import OrientationPrior
from stamp.stream.state import StreamSettings
from stamp.utils.io import resolve_output_dir
from stamp.utils.log import log_started
from stamp.utils.options.cli import NProcessesOption, OutDirOption, PickPlot3dOption, PickPlotStyleOption, PickZstackMovieOption, PlotFormatOption, PlotsOption

# Initialise Typer app
streamCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define append command
@streamCli.command()
def append(
    picks_files: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, readable=True, help='Picks files to import (.csv, .star, .json, .jsonl); one picker per file. Rows for other tomograms are ignored.'),
    ],
    tomogram: Annotated[
        Path,
        typer.Option('-r', '--tomogram', exists=True, dir_okay=False, readable=True, help='One raw tomogram (.mrc). Its file stem is the tomogram id. Unlike pick/decoy, this is a file, not a directory.', rich_help_panel = 'Inputs'),
    ],
    voxel_size_angstrom: Annotated[
        float,
        typer.Option('--voxel-size-a', help='Voxel size in Å. Fixed by the first append; checked against the MRC header.', rich_help_panel = 'Inputs'),
    ],
    box_length_angstrom: Annotated[
        float,
        typer.Option('--box-length-a', help='Subvolume box in Å (rounded to an odd voxel box). Fixed by the first append.', rich_help_panel = 'Inputs'),
    ],
    segmentation: Annotated[
        Path | None,
        typer.Option('-s', '--seg', exists=True, dir_okay=False, readable=True, help='Segmentation (.mrc) of this tomogram. A file, not a directory. Needed for normal-prior picks and membrane subtraction.', rich_help_panel = 'Inputs'),
    ] = None,
    output_dir: OutDirOption = Path('.'),
    names: Annotated[
        list[str] | None,
        typer.Option('--name', help='Picker name for each picks file, in order (repeatable). Defaults to the file stem.', rich_help_panel = 'Consensus'),
    ] = None,
    consensus_rule: Annotated[
        Literal['intersection', 'union'],
        typer.Option('--consensus-rule', help='Rule to use for consensus reconciliation.', rich_help_panel = 'Consensus'),
    ] = 'union',
    distance_threshold: Annotated[
        float,
        typer.Option('--distance-threshold', help='Voxel distance within which picks are merged.', rich_help_panel = 'Consensus'),
    ] = 15.0,
    half_set_seed: Annotated[
        int,
        typer.Option('--half-set-seed', help='Seed for half-set assignment, used by finalise.', rich_help_panel = 'Consensus'),
    ] = 0,
    orientation_prior: Annotated[
        OrientationPrior,
        typer.Option('--orientation-prior', help='Default orientation prior for picks that do not state one: normal (needs --seg), axis or none.', rich_help_panel = 'Filtering'),
    ] = OrientationPrior.NORMAL,
    max_surface_distance_angstrom: Annotated[
        float,
        typer.Option('--max-surface-distance-a', help='Picks further than this from the segmentation surface get no inferred orientation.', rich_help_panel = 'Filtering'),
    ] = 300.0,
    max_beam_angle_deviation: Annotated[
        float | None,
        typer.Option('--max-beam-angle-deviation', help='Normal deviation from beam-orthogonal plane to drop particles.', rich_help_panel = 'Filtering'),
    ] = None,
    vesicle_labels_mrc: Annotated[
        Path | None,
        typer.Option('--vesicle-labels-mrc', help='Path to one EValuator labelled MRC for this tomogram.', exists=True, dir_okay=False, readable=True, rich_help_panel = 'Filtering'),
    ] = None,
    decoy_method: Annotated[
        str,
        typer.Option('--decoy-method', help='Decoy method: auto, shifted or random-volume. synthetic-noise is not supported here (use stamp decoy).', rich_help_panel = 'Decoy'),
    ] = 'auto',
    min_shift_angstrom: Annotated[
        float,
        typer.Option('--min-shift-a', help='Minimum displacement.', rich_help_panel = 'Decoy'),
    ] = 200.0,
    max_shift_angstrom: Annotated[
        float,
        typer.Option('--max-shift-a', help='Maximum displacement.', rich_help_panel = 'Decoy'),
    ] = 600.0,
    min_distance_from_real_angstrom: Annotated[
        float,
        typer.Option('--min-distance-from-real-a', help='Minimum distance from the membrane surface.', rich_help_panel = 'Decoy'),
    ] = 100.0,
    min_distance_from_picks_angstrom: Annotated[
        float,
        typer.Option('--min-distance-from-picks-a', help='Minimum separation from any real pick.', rich_help_panel = 'Decoy'),
    ] = 60.0,
    min_border_angstrom: Annotated[
        float,
        typer.Option('--min-border-a', help='Minimum distance from the tomogram border (random-volume).', rich_help_panel = 'Decoy'),
    ] = 200.0,
    density_sign: Annotated[
        int,
        typer.Option('--density-sign', help='+1 if the membrane is brighter than background, -1 if darker.', rich_help_panel = 'Decoy'),
    ] = -1,
    decoy_seed: Annotated[
        int,
        typer.Option('--decoy-seed', help='Seed for decoy placement.', rich_help_panel = 'Decoy'),
    ] = 0,
    n_workers: NProcessesOption = 1,
    make_plots: PlotsOption = False,
    pick_plot_style: PickPlotStyleOption = 'segmented',
    plot_format: PlotFormatOption = 'tiff',
    pick_zstack_movie: PickZstackMovieOption = True,
    pick_plot_3d: PickPlot3dOption = False,
) -> None:
    '''Pick, decoy and extract one tomogram into a stream directory at <out>/stamp/stream.'''
    log_started('stream append')
    if density_sign not in (1, -1):
        raise typer.BadParameter('--density-sign must be 1 or -1.')
    picker_names = names or [path.stem for path in picks_files]
    if len(picker_names) != len(picks_files):
        raise typer.BadParameter(f'--name given {len(picker_names)} time(s) for {len(picks_files)} picks file(s)')
    # odd voxel box keeps the particle exactly centred, as classify does
    box_voxels = int(round(box_length_angstrom / voxel_size_angstrom))
    if box_voxels % 2 == 0:
        box_voxels += 1

    streamfuncs.run_stream_append(
        stream_dir=resolve_output_dir(output_dir, 'stream'),
        picks_files=picks_files,
        picker_names=picker_names,
        tomogram_path=tomogram,
        segmentation_path=segmentation,
        vesicle_labels_mrc=vesicle_labels_mrc,
        requested=StreamSettings(
            voxel_size_angstrom=voxel_size_angstrom,
            box_voxels=box_voxels,
            picker_names=picker_names,
            consensus_rule=consensus_rule,
            distance_threshold=distance_threshold,
            orientation_prior=orientation_prior.value,
            max_surface_distance_angstrom=max_surface_distance_angstrom,
            max_beam_angle_deviation=max_beam_angle_deviation,
            decoy_method=decoy_method,
            min_shift_angstrom=min_shift_angstrom,
            max_shift_angstrom=max_shift_angstrom,
            min_distance_from_real_angstrom=min_distance_from_real_angstrom,
            min_distance_from_picks_angstrom=min_distance_from_picks_angstrom,
            min_border_angstrom=min_border_angstrom,
            density_sign=density_sign,
            seed=decoy_seed,
            half_set_seed=half_set_seed,
        ),
        n_workers=n_workers,
        make_plots=make_plots,
        plot_format=plot_format,
        pick_plot_style=pick_plot_style,
        pick_zstack_movie=pick_zstack_movie,
        pick_plot_3d=pick_plot_3d,
    )

# Define finalise command
@streamCli.command()
def finalise(
    output_dir: OutDirOption = Path('.'),
) -> None:
    '''Merge appended tomograms into one real and one decoy particle set with global half-sets, then mark stream finalised.'''
    log_started('stream finalise')
    streamfuncs.run_stream_finalise(stream_dir=resolve_output_dir(output_dir, 'stream'))
