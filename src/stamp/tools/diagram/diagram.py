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
from stamp.tools.diagram import overview as overview_diagram
from stamp.tools.diagram import pick as pick_diagram
from stamp.schemas.particles import OrientationPrior
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
PriorOpt = Annotated[
    OrientationPrior,
    typer.Option('--orientation-prior', help='What is known about particle orientation: normal (membrane proteins), axis (filaments) or none.'),
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
    seed: SeedOpt = 0,
    orientation_prior: PriorOpt = OrientationPrior.NORMAL,
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
        seed=seed,
        orientation_prior=orientation_prior,
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
    field_px: FieldPxOpt = 900,
    n_vesicles: NVesiclesOpt = 14,
    seed: SeedOpt = 0,
    orientation_prior: PriorOpt = OrientationPrior.NORMAL,
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
        orientation_prior=orientation_prior,
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
    cluster_method: Annotated[
        str,
        typer.Option('--cluster-method', help='Seed clustering method: leiden, kmeans or hdbscan.'),
    ] = 'leiden',
    min_cluster_size: Annotated[
        int,
        typer.Option('--min-cluster-size', help='Smallest seed cluster that gets a reference.'),
    ] = 10,
    n_polish_rounds: Annotated[
        int,
        typer.Option('--polish-rounds', help='Polish rounds.'),
    ] = 2,
    seed: SeedOpt = 0,
    orientation_prior: PriorOpt = OrientationPrior.NORMAL,
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
        cluster_method=cluster_method,
        min_cluster_size=min_cluster_size,
        n_polish_rounds=n_polish_rounds,
        seed=seed,
        orientation_prior=orientation_prior,
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
    seed: SeedOpt = 0,
    orientation_prior: PriorOpt = OrientationPrior.NORMAL,
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
        seed=seed,
        orientation_prior=orientation_prior,
    )
