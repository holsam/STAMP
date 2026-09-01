'''
STAMP: consensus picking CLI
'''

# Import external dependencies
import json, typer
from pathlib import Path
from typing import Annotated, Literal

# Import pick command functions/variables
import stamp.commands.pick as pickfuncs

# Initialise Typer app
pickCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define pick command
@pickCli.command()
def pick(
    pickers: Annotated[
        str,
        typer.Argument(help='Comma-separated pickers to use.')
    ],
    segmentation_dir: Annotated[
        Path,
        typer.Option('-s', '--seg-dir', exists=True, file_okay=False, help='Directory of segmented MRC files; filename stems are tomogram IDs.'),
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
    picker_params: Annotated[
        str,
        typer.Option('--picker-params', help='JSON of per-picker parameters.'),
    ] = '{}',
    consensus_rule: Annotated[
        Literal['union', 'intersection'],
        typer.Option('--consensus-rule', help='Rule to use for consensus reconciliation.'),
    ] = 'union',
    distance_threshold: Annotated[
        float,
        typer.Option('--distance-threshold', help='Voxel distance within which picks are merged.'),
    ] = 15.0,
    half_set_seed: Annotated[
        int,
        typer.Option('--half-set-seed', help='Seed for half-set assignment.'),
    ] = 0,
    backend: Annotated[
        Literal['local', 'mock'],
        typer.Option('--backend', help='Backend to use for processing.'),
    ] = 'local',
) -> None:
    '''Run particle picking, reconcile across pickers, and assign half-sets.'''
    # Validate provided pickers
    picker_names = [name.strip() for name in pickers.split(',') if name.strip()]
    if not picker_names:
        raise typer.BadParameter('At least one picker must be provided.')
    for name in picker_names:
        if name not in pickfuncs.REAL_ADAPTERS:
            raise typer.BadParameter(f'Picker "{name}" not recognised.')

    # Validate picker parameters
    try:
        extra_params: dict[str, dict] = json.loads(picker_params)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f'--picker-params is not valid JSON: {exc}') from None

    # Run picking
    pickfuncs.run_pick(
        picker_names=picker_names,
        segmentation_dir=segmentation_dir,
        raw_tomogram_dir=raw_tomogram_dir,
        output_dir=output_dir,
        voxel_size_angstrom=voxel_size_angstrom,
        extra_params=extra_params,
        consensus_rule=consensus_rule,
        distance_threshold=distance_threshold,
        half_set_seed=half_set_seed,
        backend=backend,
    )
