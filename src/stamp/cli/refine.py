'''
STAMP: refinement CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import refine command functions/variables
from stamp.utils.options.cli import OutDirOption, PlotFormatOption, PlotsOption, RawDirOption, VoxelSizeOption
import stamp.commands.refine as refinefuncs
from stamp.utils.io import resolve_output_dir
from stamp.utils.log import log_started
from stamp.utils.options.literals import BackendName, PlotFormat

# Initialise Typer app
refineCli = typer.Typer(no_args_is_help=True, add_completion=False)

# Define refine command
@refineCli.command()
def refine(
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
    raw_tomogram_dir: RawDirOption,
    output_dir: OutDirOption = Path('.'),
    voxel_size_angstrom: VoxelSizeOption = None,
    threshold: Annotated[
        float,
        typer.Option('--threshold', help='FSC threshold to use for resolution calculation.')
    ] = 0.143,
    tool: Annotated[
        Literal['relion', 'm'],
        typer.Option('--tool', help='Tool to use for refinement.', rich_help_panel = 'Refinement'),
    ] = 'relion',
    mask: Annotated[
        Path | None,
        typer.Option('--mask', exists=True, dir_okay=False, rich_help_panel = 'Refinement'),
    ] = None,
    iterations: Annotated[
        int,
        typer.Option('--iterations', help='Number of refinement iterations to run.', rich_help_panel = 'Refinement'),
    ] = 5,
    backend: Annotated[
        BackendName,
        typer.Option('--backend', help='Backend to use for processing.', rich_help_panel = 'Backend'),
    ] = 'local',
    combined_halfset: Annotated[
        bool,
        typer.Option('--combined-halfset', help='Use a single combined refinement with an internal split.', rich_help_panel = 'Refinement'),
    ] = False,
    make_plots: PlotsOption = True,
    plot_format: PlotFormatOption = 'tiff',
) -> None:
    '''Refine identified classes with independent half-sets.'''
    log_started('refinement')
    refinefuncs.run_refine(
        class_id=class_id, 
        identification=identification,
        particles=particles,
        class_assignments=class_assignments,
        raw_tomogram_dir=raw_tomogram_dir,
        output_dir=resolve_output_dir(output_dir, 'refine'),
        threshold=threshold,
        tool=tool,
        mask=mask,
        iterations=iterations,
        backend=backend,
        voxel_size_angstrom=voxel_size_angstrom,
        combined_halfset=combined_halfset,
        make_plots=make_plots,
        plot_format=plot_format,
    )
