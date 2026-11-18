'''
STAMP: refinement CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import refine command functions/variables
from stamp.utils.options.cli import OptionalRawDirOption, OutDirOption, PlotFormatOption, PlotsOption, RawDirOption, VoxelSizeOption
import stamp.commands.refine as refinefuncs
from stamp.utils.io import resolve_output_dir
from stamp.utils.log import log_started

# Initialise Typer app
refineCli = typer.Typer(no_args_is_help=True, add_completion=False)

# prepare: write independent RELION half-set inputs for each identified class
@refineCli.command()
def prepare(
    class_id: Annotated[
        str,
        typer.Option('--class-id', help='Class id, or "all" for every identified class.')
    ],
    identification: Annotated[
        Path,
        typer.Option('--identification', exists=True, dir_okay=False)
    ],
    particles: Annotated[
        Path,
        typer.Option('--particles', exists=True, dir_okay=False),
    ],
    class_assignments: Annotated[
        Path,
        typer.Option('--class-assignments', exists=True, dir_okay=False),
    ],
    raw_tomogram_dir: OptionalRawDirOption = None,
    stream_dir: Annotated[
        Path | None,
        typer.Option('--stream-dir', exists=True, file_okay=False, help='Stream directory (<out>/stamp/stream).', rich_help_panel = 'Tomogram sizes'),
    ] = None,
    output_dir: OutDirOption = Path('.'),
    voxel_size_angstrom: VoxelSizeOption = None,
    coordinate_system: Annotated[
        Literal['centred', 'pixel'],
        typer.Option('--coordinate-system', help='Particle position columns: centred (RELION 5 rlnCenteredCoordinate*Angst) or pixel (rlnCoordinate*).', rich_help_panel = 'Coordinates'),
    ] = 'centred',
) -> None:
    '''Write RELION half-set inputs for an external refinement. Tomogram sizes come from the raw headers in -r/--raw-dir or from --stream-dir.'''
    log_started('refinement preparation')
    refinefuncs.run_refine_prepare(
        class_id=class_id,
        identification=identification,
        particles=particles,
        class_assignments=class_assignments,
        raw_tomogram_dir=raw_tomogram_dir,
        output_dir=resolve_output_dir(output_dir, 'refine'),
        voxel_size_angstrom=voxel_size_angstrom,
        coordinate_system=coordinate_system,
        stream_dir=stream_dir,
    )

# assess: FSC of two externally refined half-maps for one class
@refineCli.command()
def assess(
    class_id: Annotated[
        str,
        typer.Option('--class-id', help='Class id the half-maps belong to.')
    ],
    prepared_dir: Annotated[
        Path,
        typer.Option('--prepared-dir', exists=True, file_okay=False, help='Class directory written by refine prepare.')
    ],
    half_a: Annotated[
        Path,
        typer.Option('--half-a', exists=True, dir_okay=False, help='Final map of the half A refinement.')
    ],
    half_b: Annotated[
        Path,
        typer.Option('--half-b', exists=True, dir_okay=False, help='Final map of the half B refinement.')
    ],
    threshold: Annotated[
        float,
        typer.Option('--threshold', help='FSC threshold to use for resolution calculation.')
    ] = 0.143,
    mask: Annotated[
        Path | None,
        typer.Option('--mask', exists=True, dir_okay=False, rich_help_panel='Refinement'),
    ] = None,
    make_plots: PlotsOption = True,
    plot_format: PlotFormatOption = 'tiff',
) -> None:
    '''Assess two externally refined half-maps for one class.'''
    log_started('refinement assessment')
    refinefuncs.run_refine_assess(
        class_id=class_id,
        prepared_dir=prepared_dir,
        half_a=half_a,
        half_b=half_b,
        threshold=threshold,
        mask=mask,
        make_plots=make_plots,
        plot_format=plot_format,
    )
