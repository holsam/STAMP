'''
STAMP: unsupervised classification CLI
'''

# Import external dependencies
import json, typer
from pathlib import Path
from typing import Annotated, Literal

# Import classify command functions/variables
import stamp.commands.classify as classifyfuncs
from stamp.decoy.validate import is_decoy_particle_set
from stamp.schemas.particles import ParticleSet
from stamp.utils.errors import StampPipelineError
from stamp.utils.io import resolve_output_dir

# Initialise Typer app
classifyCli = typer.Typer(
    no_args_is_help = True,
    add_completion = False,
)

# Define classify command
@classifyCli.command()
def classify(
    particles: Annotated[
        Path,
        typer.Option('-p', '--particles', help='particle_set.json or decoy_particle_set.json', exists=True)
    ],
    raw_tomogram_dir: Annotated[
        Path,
        typer.Option('-r', '--raw-dir', help='Raw tomogram directory.', exists=True, file_okay=False),
    ],
    voxel_size_angstrom: Annotated[
        float | None,
        typer.Option('--voxel-size-a', help='Voxel size in Å. Read from MRC headers if omitted.'),
    ] = None,
    segmentation_dir: Annotated[
        Path | None,
        typer.Option('-s', '--seg-dir', help='Segmentation directory for membrane voxel replacement.', exists=True, file_okay=False),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option('-o', '--out-dir', file_okay=False, help='Output directory.'),
    ] = Path('.'),
    box_angstrom: Annotated[
        float,
        typer.Option('--box-length-a', help='Extraction box edge length, in Å.', rich_help_panel = 'Subvolume extraction')
    ] = 300.0,
    n_radial_bins: Annotated[
        int,
        typer.Option('--n-bins', help='Radial bins in the rotational average.', rich_help_panel = 'Subvolume extraction'),
    ] = 12,
    method: Annotated[
        Literal['hbdscan', 'kmeans'],
        typer.Option('--method', help='Clustering method to use.', rich_help_panel = 'Clustering'),
    ] = 'hbdscan',
    min_cluster_size: Annotated[
        int,
        typer.Option('--min-cluster-size', help='Minimum cluster size (used by HDBSCAN).', rich_help_panel = 'Clustering'),
    ] = 20,
    n_clusters: Annotated[
        int,
        typer.Option('--n-clusters', help='Number of clusters (used by KMeans).', rich_help_panel = 'Clustering'),
    ] = 5,
    n_components: Annotated[
        int,
        typer.Option('--n-components', help='Number of PCA components.', rich_help_panel = 'Clustering'),
    ] = 20,
    strict_halfset_independence: Annotated[
        bool,
        typer.Option('--strict-halfset-independence', help='Cluster each half separately and match clusters afterwards.', rich_help_panel = 'Clustering'),
    ] = True,
    random_state: Annotated[
        int,
        typer.Option('--seed', help='Seed for PCA and KMeans.', rich_help_panel = 'Clustering'),
    ] = 0,
    inplane_alignment: Annotated[
        bool,
        typer.Option('--inplane-alignment/--no-inplane-alignment', help='Estimate per-particle in-plane angle for averaging.', rich_help_panel = 'In-plane alignment'),
    ] = True,
    inplane_angular_step_degrees: Annotated[
        float, typer.Option('--inplane-step-deg', help='In-plane search step in degrees.', rich_help_panel = 'In-plane alignment'),
    ] = 10.0,
    inplane_iterations: Annotated[
        int, typer.Option('--inplane-iterations', help='Number of reference refinement rounds to run.', rich_help_panel = 'In-plane alignment'),
    ] = 3,
    azimuthal_modes: Annotated[
        int,
        typer.Option(help='Highest azimuthal Fourier mode retained. 0 reproduces pure rotational averaging; 4 captures C4 symmetry. Higher modes are increasingly noisy.', rich_help_panel = 'Subvolume extraction'),
    ] = 4,
    min_radius_fraction: Annotated[
        float,
        typer.Option(help='Radial bins below this fraction of the box radius are excluded from azimuthal features (too few voxels to report symmetry).', rich_help_panel = 'Subvolume extraction'),
    ] = 0.25,
    n_azimuthal_samples: Annotated[
        int,
        typer.Option(help='Azimuthal sampling points (must be at least 2*(modes+1)).', rich_help_panel = 'Subvolume extraction'),
    ] = 64,
    n_workers: Annotated[
        int,
        typer.Option('-n', '--n-processes', help='Number of processes to use for per-cluster in-plane alignment (1 = sequential).'),
    ] = 1,
    make_plots: Annotated[
        bool,
        typer.Option('--plots/--no-plots', help='Write embedding and class-average plots.', rich_help_panel = 'Plotting'),
    ] = True,
    plot_format: Annotated[
        Literal['png', 'jpg', 'tiff', 'svg'],
        typer.Option('--plot-format', help='Image format for static plots.', rich_help_panel = 'Plotting'),
    ] = 'tiff',
    keep_raw: Annotated[
        bool,
        typer.Option('--keep-raw', help='Keep the raw per-tomogram subvolume cache instead of archiving it to raw.tar.gz.'),
    ] = False,
) -> None:
    '''Cluster picked particles by structural similarity.'''
    is_decoy = is_decoy_particle_set(ParticleSet.model_validate(json.loads(particles.read_text())))
    classifyfuncs.run_classify(
        particles=particles,
        raw_tomogram_dir=raw_tomogram_dir,
        segmentation_dir=segmentation_dir,
        output_dir=resolve_output_dir(output_dir, 'classify', 'decoy' if is_decoy else None),
        voxel_size_angstrom=voxel_size_angstrom,
        box_angstrom=box_angstrom,
        n_radial_bins=n_radial_bins,
        method=method,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        n_components=n_components,
        strict_halfset_independence=strict_halfset_independence,
        random_state=random_state,
        inplane_alignment=inplane_alignment,
        inplane_angular_step_degrees=inplane_angular_step_degrees,
        inplane_iterations=inplane_iterations,
        n_workers=n_workers,
        azimuthal_modes=azimuthal_modes,
        min_radius_fraction=min_radius_fraction,
        n_azimuthal_samples=n_azimuthal_samples,
        make_plots=make_plots,
        plot_format=plot_format,
        keep_raw=keep_raw,
    )
