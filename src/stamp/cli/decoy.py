'''
STAMP: decoy dataset generation CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import decoy command functions/variables
import stamp.commands.decoy as decoyfuncs
from stamp.utils.errors import StampPipelineError
from stamp.utils.io import resolve_output_dir
from stamp.utils.log import log_started

# Initialise Typer app
decoyCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define decoy command
@decoyCli.command()
def decoy(
    voxel_size_angstrom: Annotated[
        float | None,
        typer.Option('--voxel-size-a', help='Voxel size in Å. Read from MRC headers if omitted.'),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option('-o', '--out-dir', file_okay=False, help='Output directory.'),
    ] = Path('.'),
    method: Annotated[
        Literal['shifted', 'synthetic-noise', 'random-volume'],
        typer.Option('--method', help='Decoy method to use.'),
    ] = 'shifted',
    real_particle_set: Annotated[
        Path | None,
        typer.Option('--real-particle-set', help='particle_set.json from `stamp pick`. Required unless using --method synthetic-noise.', rich_help_panel = 'Inputs'),
    ] = None,
    segmentation_dir: Annotated[
        Path | None,
        typer.Option('-s', '--seg-dir', help='Segmentation directory. Required unless using --method synthetic-noise.', rich_help_panel = 'Inputs'),
    ] = None,
    raw_tomogram_dir: Annotated[
        Path | None,
        typer.Option('-r', '--raw-dir', help='Raw tomogram directory. Required unless using --method synthetic-noise.', rich_help_panel = 'Inputs'),
    ] = None,
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
    n_workers: Annotated[
        int,
        typer.Option('-n', '--n-processes', help='Number of processes to use for per-tomogram decoy generation (1 = sequential).', rich_help_panel = 'Decoy placement'),
    ] = 1,
    keep_raw: Annotated[
        bool,
        typer.Option('--keep-raw', help='Keep the raw per-picker output directory instead of archiving it to raw.tar.gz.'),
    ] = False,
    make_plots: Annotated[
        bool,
        typer.Option('--plots', help='Write decoy position plots.', rich_help_panel = 'Plotting'),
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
) -> None:
    '''Generate a decoy dataset to run through STAMP alongside real data.'''
    log_started('decoy generation')
    if density_sign not in (1, -1):
        raise typer.BadParameter('--density-sign must be 1 or -1.')

    # Validate method requirements
    if method == 'random-volume':
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
