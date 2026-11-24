'''
STAMP: CLI for `stamp tools diagram` command group
'''

# Import external dependencies
import typer
from pathlib import Path
from types import ModuleType
from typing import Annotated

# Import diagram stage renderers
from stamp.tools.diagram import classify as classify_diagram
from stamp.tools.diagram import identify as identify_diagram
from stamp.tools.diagram import overview as overview_diagram
from stamp.tools.diagram import pick as pick_diagram
from stamp.tools.diagram import stream as stream_diagram
from enum import Enum
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
# DiagramPrior: the orientation priors, or mixed for one tomogram holding all three
class DiagramPrior(str, Enum):
    NORMAL = 'normal'
    AXIS = 'axis'
    NONE = 'none'
    MIXED = 'mixed'

PriorOpt = Annotated[
    DiagramPrior,
    typer.Option('--orientation-prior', help='What is known about particle orientation: normal (membrane proteins), axis (filaments), none, or mixed (one tomogram with all three, each group classified on its own; not for overview).'),
]
AllPriorsOpt = Annotated[
    bool,
    typer.Option('--all-priors', help='Render once per orientation prior into <out-dir>/<prior>/ and add a comparison figure (--orientation-prior is ignored).'),
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

# _run: build the config and hand it to a stage renderer module, which renders one prior or all of them
def _run(
    renderer: ModuleType,
    target: Path,
    distractor: Path,
    also: list[Path] | None,
    output_dir: Path,
    **overrides,
) -> None:
    all_priors = overrides.pop('all_priors')
    prior = overrides.pop('orientation_prior')
    if prior is DiagramPrior.MIXED:
        if all_priors or not hasattr(renderer, 'render_mixed_priors'):
            raise typer.BadParameter('mixed is for the pick, classify and identify diagrams and cannot be combined with --all-priors', param_hint='--orientation-prior')
        renderer.render_mixed_priors(_config(target, distractor, also, output_dir, **overrides))
        return
    overrides['orientation_prior'] = OrientationPrior(prior.value)
    (renderer.render_all if all_priors else renderer.render)(_config(target, distractor, also, output_dir, **overrides))

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
    orientation_prior: PriorOpt = DiagramPrior.NORMAL,
    all_priors: AllPriorsOpt = False,
) -> None:
    '''Generate a diagram of the overall STAMP workflow.'''
    _run(
        overview_diagram,
        target,
        distractor,
        also,
        output_dir,
        n_synthetic_unknowns=n_synthetic_unknowns,
        box_size=box_size,
        pixel_size=pixel_size,
        seed=seed,
        orientation_prior=orientation_prior,
        all_priors=all_priors,
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
    orientation_prior: PriorOpt = DiagramPrior.NORMAL,
    all_priors: AllPriorsOpt = False,
) -> None:
    '''Generate a diagram of the pick command workflow.'''
    _run(
        pick_diagram,
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
        all_priors=all_priors,
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
    orientation_prior: PriorOpt = DiagramPrior.NORMAL,
    all_priors: AllPriorsOpt = False,
) -> None:
    '''Generate a diagram of the classify command workflow.'''
    _run(
        classify_diagram,
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
        all_priors=all_priors,
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
    orientation_prior: PriorOpt = DiagramPrior.NORMAL,
    all_priors: AllPriorsOpt = False,
) -> None:
    '''Generate a diagram of the identify command workflow.'''
    _run(
        identify_diagram,
        target,
        distractor,
        also,
        output_dir,
        n_synthetic_unknowns=n_synthetic_unknowns,
        box_size=box_size,
        pixel_size=pixel_size,
        seed=seed,
        orientation_prior=orientation_prior,
        all_priors=all_priors,
    )

# Define stream diagram command
@diagramCli.command('stream', rich_help_panel='Command Diagrams')
def stream_subcommand(
    target: TargetArg,
    distractor: DistractorArg,
    output_dir: OutDirOpt = Path('out/diagrams/stream'),
    also: AlsoOpt = None,
    n_synthetic_unknowns: UnknownsOpt = 2,
    box_size: BoxOpt = 64,
    pixel_size: PixelOpt = 3.0,
    n_tomograms: Annotated[int, typer.Option('--n-tomograms', min=2, help='Tomograms appended before finalise.')] = 4,
    seed: SeedOpt = 0,
    orientation_prior: PriorOpt = DiagramPrior.NORMAL,
) -> None:
    '''Generate a diagram of the stream workflow.'''
    _run(
        stream_diagram,
        target,
        distractor,
        also,
        output_dir,
        n_synthetic_unknowns=n_synthetic_unknowns,
        box_size=box_size,
        pixel_size=pixel_size,
        n_tomograms=n_tomograms,
        seed=seed,
        orientation_prior=orientation_prior,
        all_priors=False,
    )
