'''
STAMP: decoy dataset generation CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import decoy command functions/variables
from stamp.utils.options.cli import KeepRawOption, NProcessesOption, OptionalRawDirOption, OutDirOption, PickPlot3dOption, PickPlotStyleOption, PickZstackMovieOption, PlotFormatOption, PlotsOption, SegDirOption, VoxelSizeOption
import stamp.commands.decoy as decoyfuncs
from stamp.utils.errors import StampPipelineError
from stamp.utils.io import resolve_output_dir
from stamp.utils.log import log_started
from stamp.utils.options.literals import PlotFormat

# Initialise Typer app
decoyCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define decoy command
@decoyCli.command()
def decoy(
    voxel_size_angstrom: VoxelSizeOption = None,
    output_dir: OutDirOption = Path('.'),
    method: Annotated[
        Literal['auto', 'shifted', 'synthetic-noise', 'random-volume'],
        typer.Option('--method', help='Decoy method to use (auto: resolve per-prior).'),
    ] = 'auto',
    real_particle_set: Annotated[
        Path | None,
        typer.Option('--real-particle-set', help='particle_set.json from `stamp pick`. Required unless using --method synthetic-noise.', rich_help_panel = 'Inputs'),
    ] = None,
    segmentation_dir: SegDirOption = None,
    raw_tomogram_dir: OptionalRawDirOption = None,
    density_sign: Annotated[
        int,
        typer.Option('--density-sign', help='+1 if the synthetic membrane shell is brighter than background, -1 if darker.', rich_help_panel = 'Synthetic noise decoy'),
    ] = -1,
    n_decoys_per_tomogram: Annotated[
        int | None,
        typer.Option('--n-decoys-per-tomogram', help='Decoy positions per noise volume. Required for --method synthetic-noise; shifted places one decoy per real pick.', rich_help_panel = 'Decoy placement'),
    ] = None,
    min_distance_from_real_angstrom: Annotated[
        float,
        typer.Option('--min-distance-from-real-a', help='Minimum distance from the membrane surface.', rich_help_panel = 'Shifted decoy'),
    ] = 100.0,
    min_distance_from_picks_angstrom: Annotated[
        float,
        typer.Option('--min-pick-distance-a', help='Minimum separation from any real pick.', rich_help_panel = 'Shifted decoy'),
    ] = 60.0,
    min_shift_angstrom: Annotated[
        float,
        typer.Option('--min-shift-a', help='Minimum displacement.', rich_help_panel = 'Shifted decoy'),
    ] = 200.0,
    max_shift_angstrom: Annotated[
        float,
        typer.Option('--max-shift-a', help='Maximum displacement.', rich_help_panel = 'Shifted decoy'),
    ] = 600.0,
    min_border_angstrom: Annotated[
        float,
        typer.Option('--min-border-a', help='Minimum distance from the tomogram border.', rich_help_panel = 'Random volume decoy'),
    ] = 200.0,
    n_synthetic_tomograms: Annotated[
        int,
        typer.Option('--n-synthetic-tomograms', help='Noise volumes.', rich_help_panel = 'Synthetic noise decoy'),
    ] = 3,
    synthetic_shape: Annotated[
        str,
        typer.Option('--synthetic-shape', help='z,y,x dimensions of volume to create.', rich_help_panel = 'Synthetic noise decoy'),
    ] = '200,200,200',
    seed: Annotated[
        int,
        typer.Option('--seed', help='Seed for sampling and half-set assignment.', rich_help_panel = 'Decoy placement'),
    ] = 0,
    n_workers: NProcessesOption = 1,
    keep_raw: KeepRawOption = False,
    make_plots: PlotsOption = False,
    pick_plot_style: PickPlotStyleOption = 'segmented',
    plot_format: PlotFormatOption = 'tiff',
    pick_zstack_movie: PickZstackMovieOption = True,
    pick_plot_3d: PickPlot3dOption = False,
) -> None:
    '''Generate a decoy dataset to run through STAMP alongside real data.'''
    log_started('decoy generation')
    if density_sign not in (1, -1):
        raise typer.BadParameter('--density-sign must be 1 or -1.')

    # Validate method requirements
    if method == 'auto':
        if any([real_particle_set is None, raw_tomogram_dir is None]):
            raise StampPipelineError('--method auto requires --real-particle-set and -r/--raw-dir')
    elif method == 'random-volume':
        if any([real_particle_set is None, raw_tomogram_dir is None]):
            raise StampPipelineError('--method random-volume requires --real-particle-set and -r/--raw-dir')
    elif method == 'shifted':
        if any([real_particle_set is None, segmentation_dir is None, raw_tomogram_dir is None]):
            raise StampPipelineError(f'--method {method} requires --real-particle-set, -s/--seg-dir and -r/--raw--dir')
    else:
        if all([voxel_size_angstrom is None, segmentation_dir is None, raw_tomogram_dir is None]):
            raise StampPipelineError(f'--method {method} requires one of --voxel-size-a, -s/--seg-dir or -r/--raw-dir to be set.')

    # Run decoy generation
    decoyfuncs.run_decoy(
        output_dir=resolve_output_dir(output_dir, 'decoy'),
        method=method,
        real_particle_set=real_particle_set,
        segmentation_dir=segmentation_dir,
        raw_tomogram_dir=raw_tomogram_dir,
        voxel_size_angstrom=voxel_size_angstrom,
        n_decoys_per_tomogram=n_decoys_per_tomogram,
        min_distance_from_real_angstrom=min_distance_from_real_angstrom,
        min_distance_from_picks_angstrom=min_distance_from_picks_angstrom,
        min_shift_angstrom=min_shift_angstrom,
        max_shift_angstrom=max_shift_angstrom,
        n_synthetic_tomograms=n_synthetic_tomograms,
        synthetic_shape=synthetic_shape,
        seed=seed,
        density_sign=density_sign,
        n_workers=n_workers,
        keep_raw=keep_raw,
        make_plots=make_plots,
        pick_plot_style=pick_plot_style,
        plot_format=plot_format,
        pick_zstack_movie=pick_zstack_movie,
        pick_plot_3d=pick_plot_3d,
    )
