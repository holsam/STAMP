'''
STAMP: CLI for `stamp tools diagram` command group
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Callable

# Import diagram stage renderers
from stamp.tools.diagram import classify as classify_diagram
from stamp.tools.diagram import identify as identify_diagram
from stamp.tools.diagram import native as native_diagram
from stamp.tools.diagram import overview as overview_diagram
from stamp.tools.diagram import pick as pick_diagram
from stamp.tools.diagram.utils.config import DiagramConfig

# Initialise Typer app
diagramCli = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
)

# Shared option / argument annotations
TargetArg = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, help='Example protein PDB to use.'),
]
DistractorArg = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, help='PDB of a known different protein.'),
]
AlsoOpt = Annotated[
    list[Path] | None,
    typer.Option('--also', help='Extra PDBs which are not included in the identify search panel.'),
]
OutDirOpt = Annotated[
    Path,
    typer.Option('-o', '--out-dir', file_okay=False, help='Output directory for the PNGs.'),
]
UnknownsOpt = Annotated[
    int,
    typer.Option('--n-synthetic-unknowns', help='Procedurally generated extra classes with no PDB.'),
]
BoxOpt = Annotated[
    int,
    typer.Option('--box-size-px', help='Particle box edge length, in pixels.'),
]
PixelOpt = Annotated[
    float,
    typer.Option('--pixel-size-a', help='Angstroms per pixel.'),
]
SeedOpt = Annotated[
    int,
    typer.Option('--seed', help='Random seed.'),
]
FieldPxOpt = Annotated[
    int,
    typer.Option('--field-px', help='Side of the mock tomogram slice, pixels.'),
]
NVesiclesOpt = Annotated[
    int,
    typer.Option('--n-vesicles', help='Vesicles in the mock scene.'),
]
IdentifyMinCcOpt = Annotated[
    float,
    typer.Option('--identify-min-cc', help='Minimum CC to call a class.'),
]
IdentifyMinGapOpt = Annotated[
    float,
    typer.Option('--identify-min-gap', help='Minimum CC gap to runner-up.'),
]

# _config: assemble a DiagramConfig, leaving unspecified knobs at their defaults
def _config(
    target: Path,
    distractor: Path,
    also: list[Path] | None,
    output_dir: Path,
    **overrides,
) -> DiagramConfig:
    return DiagramConfig(
        target=target,
        distractor=distractor,
        also=list(also) if also else [],
        outdir=output_dir,
        **overrides,
    )

# _run: build the config and hand it to a stage renderer
def _run(
    renderer: Callable[[DiagramConfig], object],
    target: Path,
    distractor: Path,
    also: list[Path] | None,
    output_dir: Path,
    **overrides,
) -> None:
    renderer(_config(target, distractor, also, output_dir, **overrides))

# Define overview diagram command
@diagramCli.command('overview', rich_help_panel='Overview Diagrams')
def overview_subcommand(
    target: TargetArg,
    distractor: DistractorArg,
    output_dir: OutDirOpt = Path('out/diagrams/overview'),
    also: AlsoOpt = None,
    n_synthetic_unknowns: UnknownsOpt = 2,
    box_size: BoxOpt = 64,
    pixel_size: PixelOpt = 3.0,
    identify_min_cc: IdentifyMinCcOpt = 0.72,
    identify_min_gap: IdentifyMinGapOpt = 0.06,
    seed: SeedOpt = 0,
) -> None:
    '''Generate a diagram of the overall STAMP workflow.'''
    _run(
        overview_diagram.render,
        target,
        distractor,
        also,
        output_dir,
        n_synthetic_unknowns=n_synthetic_unknowns,
        box_size=box_size,
        pixel_size=pixel_size,
        identify_min_cc=identify_min_cc,
        identify_min_gap=identify_min_gap,
        seed=seed,
    )

# Define pick diagram command
@diagramCli.command('pick', rich_help_panel='Command Diagrams')
def pick_subcommand(
    target: TargetArg,
    distractor: DistractorArg,
    output_dir: OutDirOpt = Path('out/diagrams/pick'),
    also: AlsoOpt = None,
    n_synthetic_unknowns: UnknownsOpt = 2,
    box_size: BoxOpt = 64,
    pixel_size: PixelOpt = 3.0,
    field_px: FieldPxOpt = 760,
    n_vesicles: NVesiclesOpt = 10,
    seed: SeedOpt = 0,
) -> None:
    '''Generate a diagram of the pick command workflow.'''
    _run(
        pick_diagram.render,
        target,
        distractor,
        also,
        output_dir,
        n_synthetic_unknowns=n_synthetic_unknowns,
        box_size=box_size,
        pixel_size=pixel_size,
        field_px=field_px,
        n_vesicles=n_vesicles,
        seed=seed,
    )

# Define native diagram command
@diagramCli.command('native', rich_help_panel='Command Diagrams')
def native_subcommand(
    target: TargetArg,
    distractor: DistractorArg,
    output_dir: OutDirOpt = Path('out/diagrams/native'),
    also: AlsoOpt = None,
    n_synthetic_unknowns: UnknownsOpt = 2,
    box_size: BoxOpt = 64,
    pixel_size: PixelOpt = 3.0,
    field_px: FieldPxOpt = 760,
    n_vesicles: NVesiclesOpt = 10,
    offset_min_a: Annotated[
        float,
        typer.Option('--offset-min-a', help='Shell inner offset from the membrane, in Å.'),
    ] = 20.0,
    offset_max_a: Annotated[
        float,
        typer.Option('--offset-max-a', help='Shell outer offset from the membrane, in Å.'),
    ] = 80.0,
    n_samples: Annotated[
        int,
        typer.Option('--n-samples', help='Samples along each normal.'),
    ] = 5,
    surface_spacing_a: Annotated[
        float,
        typer.Option('--surface-spacing-a', help='Surface point spacing, in Å.'),
    ] = 20.0,
    min_particle_distance_a: Annotated[
        float,
        typer.Option('--min-particle-distance-a', help='NMS minimum separation, in Å.'),
    ] = 60.0,
    n_mad: Annotated[
        float,
        typer.Option('--n-mad', help='Robust-score threshold, in MAD units.'),
    ] = 3.0,
    density_sign: Annotated[
        int,
        typer.Option('--density-sign', help='+1: protein brighter than background (this mock); -1: protein darker (real cryo-ET).'),
    ] = 1,
    seed: SeedOpt = 0,
) -> None:
    '''Generate a diagram of the template-free native picker workflow.'''
    _run(
        native_diagram.render,
        target,
        distractor,
        also,
        output_dir,
        n_synthetic_unknowns=n_synthetic_unknowns,
        box_size=box_size,
        pixel_size=pixel_size,
        field_px=field_px,
        n_vesicles=n_vesicles,
        offset_min_a=offset_min_a,
        offset_max_a=offset_max_a,
        n_samples=n_samples,
        surface_spacing_a=surface_spacing_a,
        min_particle_distance_a=min_particle_distance_a,
        n_mad=n_mad,
        density_sign=density_sign,
        seed=seed,
    )

# Define classify diagram command
@diagramCli.command('classify', rich_help_panel='Command Diagrams')
def classify_subcommand(
    target: TargetArg,
    distractor: DistractorArg,
    output_dir: OutDirOpt = Path('out/diagrams/classify'),
    also: AlsoOpt = None,
    n_synthetic_unknowns: UnknownsOpt = 2,
    box_size: BoxOpt = 64,
    pixel_size: PixelOpt = 3.0,
    kmeans_k: Annotated[
        int | None,
        typer.Option('--kmeans-k', help='KMeans clusters (default: n species + 1).'),
    ] = None,
    hdbscan_min_cluster_size: Annotated[
        int,
        typer.Option('--hdbscan-min-cluster-size', help='HDBSCAN min cluster size.'),
    ] = 15,
    seed: SeedOpt = 0,
) -> None:
    '''Generate a diagram of the classify command workflow.'''
    _run(
        classify_diagram.render,
        target, distractor,
        also,
        output_dir,
        n_synthetic_unknowns=n_synthetic_unknowns,
        box_size=box_size,
        pixel_size=pixel_size,
        kmeans_k=kmeans_k,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        seed=seed,
    )

# Define identify diagram command
@diagramCli.command('identify', rich_help_panel='Command Diagrams')
def identify_subcommand(
    target: TargetArg,
    distractor: DistractorArg,
    output_dir: OutDirOpt = Path('out/diagrams/identify'),
    also: AlsoOpt = None,
    n_synthetic_unknowns: UnknownsOpt = 2,
    box_size: BoxOpt = 64,
    pixel_size: PixelOpt = 3.0,
    identify_min_cc: IdentifyMinCcOpt = 0.72,
    identify_min_gap: IdentifyMinGapOpt = 0.06,
    decoy_margin: Annotated[
        float, typer.Option('--decoy-margin', help='Minimum CC margin over the decoy model.'),
    ] = 0.05,
    seed: SeedOpt = 0,
) -> None:
    '''Generate a diagram of the identify command workflow.'''
    _run(
        identify_diagram.render,
        target,
        distractor,
        also,
        output_dir,
        n_synthetic_unknowns=n_synthetic_unknowns,
        box_size=box_size,
        pixel_size=pixel_size,
        identify_min_cc=identify_min_cc,
        identify_min_gap=identify_min_gap,
        decoy_margin=decoy_margin,
        seed=seed,
    )
