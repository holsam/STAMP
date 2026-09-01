'''
STAMP: decoy dataset generation CLI
'''

# Import external dependencies
import json, typer
from pathlib import Path
from typing import Annotated, Literal

# Import decoy command functions/variables
import stamp.commands.decoy as decoyfuncs

# Initialise Typer app
decoyCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define decoy command
@decoyCli.command()
def decoy(
    output_dir: Annotated[
        Path,
        typer.Option('-o', '--out-dir', file_okay=False, help='Output directory.'),
    ],
    voxel_size_angstrom: Annotated[
        float,
        typer.Option('--voxel-size-a', help='Voxel size, in Ångstrom.'),
    ],
    method: Annotated[
        Literal['rejected-surface', 'shifted', 'synthetic-noise'],
        typer.Option('--method', help='Decoy method to use.'),
    ] = 'rejected-surface',
    real_particle_set: Annotated[
        Path | None,
        typer.Option('--real-particle-set', help='particle_set.json from `stamp pick`. Required unless using --method synthetic-noise.'),
    ] = None,
    segmentation_dir: Annotated[
        Path | None,
        typer.Option('-s', '--seg-dir', help='Segmentation directory. Required unless using --method synthetic-noise.'),
    ] = None,
    raw_tomogram_dir: Annotated[
        Path | None,
        typer.Option('-r', '--raw-dir', help='Raw tomogram directory. Required unless using --method synthetic-noise.'),
    ] = None,
    picker_params: Annotated[
        str,
        typer.Option('--picker-params', help='JSON of native-picker parameters. Must match the values used for `stamp pick`, or decoys won\'t be drawn from the same candidate pool.'),
    ] = '{}',
    n_decoys_per_tomogram: Annotated[
        int,
        typer.Option('--n-decoys-per-tomogram', help='Decoy positions per tomogram.'),
    ] = 50,
    min_distance_from_real_angstrom: Annotated[
        float,
        typer.Option('--min-distance-from-real-a', help='Minimum separation from any real pick (rejected-surface).'),
    ] = 100.0,
    min_shift_angstrom: Annotated[
        float,
        typer.Option('--min-shift-a', help='Minimum displacement (shifted).'),
    ] = 200.0,
    max_shift_angstrom: Annotated[
        float,
        typer.Option('--max-shift-a', help='Maximum displacement (shifted).'),
    ] = 600.0,
    n_synthetic_tomograms: Annotated[
        int,
        typer.Option('--n-synthetic-tomograms', help='Noise volumes (synthetic-noise).'),
    ] = 3,
    synthetic_shape: Annotated[
        str,
        typer.Option('--synthetic-shape', help='z,y,x (synthetic-noise).'),
    ] = '200,200,200',
    seed: Annotated[
        int,
        typer.Option('--seed', help='Seed for sampling and half-set assignment.'),
    ] = 0,
) -> None:
    '''Generate a decoy dataset to run through STAMP alongside real data.'''
    # Validate picker parameters
    try:
        parameters = json.loads(picker_params)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f'--picker-params is not valid JSON: {exc}') from None

    # Run decoy generation
    decoyfuncs.run_decoy(
        output_dir=output_dir,
        method=method,
        parameters=parameters,
        real_particle_set=real_particle_set,
        segmentation_dir=segmentation_dir,
        raw_tomogram_dir=raw_tomogram_dir,
        voxel_size_angstrom=voxel_size_angstrom,
        n_decoys_per_tomogram=n_decoys_per_tomogram,
        min_distance_from_real_angstrom=min_distance_from_real_angstrom,
        min_shift_angstrom=min_shift_angstrom,
        max_shift_angstrom=max_shift_angstrom,
        n_synthetic_tomograms=n_synthetic_tomograms,
        synthetic_shape=synthetic_shape,
        seed=seed,
    )
