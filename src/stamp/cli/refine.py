'''
STAMP: refinement CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

import stamp.commands.refine as refinefuncs

refineCli = typer.Typer(no_args_is_help=True, add_completion=False)

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
    raw_tomogram_dir: Annotated[
        Path,
        typer.Option('-r', '--raw-dir', exists=True, file_okay=False, help='Directory of raw tomogram MRCs, matched to segmentations by filename stem.'),
    ],
    output_dir: Annotated[
        Path,
        typer.Option('-o', '--out-dir', file_okay=False, help='Directory to write outputs to.'),
    ] = Path('.'),
    voxel_size_angstrom: Annotated[
        float | None,
        typer.Option('--voxel-size-a', help='Voxel size of the input tomograms, in Angstrom.'),
    ] = None,
    tool: Annotated[
        Literal['relion', 'm'],
        typer.Option('--tool'),
    ] = 'relion',
    mask: Annotated[
        Path | None,
        typer.Option('--mask', exists=True, dir_okay=False),
    ] = None,
    iterations: Annotated[
        int,
        typer.Option('--iterations'),
    ] = 5,
    backend: Annotated[
        Literal['local', 'mock'],
        typer.Option('--backend', help='Backend to use for processing.'),
    ] = 'local',
    combined_halfset: Annotated[
        bool, 
        typer.Option('--combined-halfset', help='Use a single combined refinement with an internal split.'),
    ] = False,
) -> None:
    '''Refine identified classes with independent half-sets.'''
    refinefuncs.run_refine(
        class_id=class_id, 
        identification=identification,
        particles=particles,
        class_assignments=class_assignments,
        raw_tomogram_dir=raw_tomogram_dir,
        output_dir=output_dir,
        tool=tool,
        mask=mask,
        iterations=iterations,
        backend=backend,
        voxel_size_angstrom=voxel_size_angstrom,
        combined_halfset=combined_halfset,
    )
