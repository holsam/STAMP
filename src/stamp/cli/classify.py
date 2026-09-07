'''
STAMP: unsupervised classification CLI
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import classify command functions/variables
import stamp.commands.classify as classifyfuncs

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
        typer.Option(help='particle_set.json or decoy_particle_set.json', exists=True)
    ],
    raw_tomogram_dir: Annotated[
        Path,
        typer.Option('-r', '--raw-dir', help='Raw tomogram directory.', exists=True, file_okay=False),
    ],
    output_dir: Annotated[
        Path,
        typer.Option('-o', '--out-dir', file_okay=False, help='Output directory.'),
    ],
    voxel_size_angstrom: Annotated[
        float,
        typer.Option('--voxel-size-a', help='Voxel size, in Å.'),
    ],
    segmentation_dir: Annotated[
        Path | None,
        typer.Option('-s', '--seg-dir', help='Segmentation directory for membrane voxel replacement.', exists=True, file_okay=False),
    ] = None,
    box_angstrom: Annotated[
        float,
        typer.Option('--box-length-a', help='Extraction box edge length, in Å.')
    ] = 300.0,
    n_radial_bins: Annotated[
        int,
        typer.Option('--n-bins', help='Radial bins in the rotational average.'),
    ] = 12,
    method: Annotated[
        Literal['hbdscan', 'kmeans'],
        typer.Option('--method', help='Clustering method to use.'),
    ] = 'hbdscan',
    min_cluster_size: Annotated[
        int,
        typer.Option('--min-cluster-size', help='Minimum cluster size (used by HDBSCAN).'),
    ] = 20,
    n_clusters: Annotated[
        int,
        typer.Option('--n-clusters', help='Number of clusters (used by KMeans).'),
    ] = 5,
    n_components: Annotated[
        int,
        typer.Option('--n-components', help='Number of PCA components.'),
    ] = 20,
    strict_halfset_independence: Annotated[
        bool,
        typer.Option('--strict-halfset-independence', help='Cluster each half separately and match clusters afterwards.'),
    ] = False,
    random_state: Annotated[
        int,
        typer.Option('--seed', help='Seed for PCA and KMeans.'),
    ] = 0,
) -> None:
    '''Cluster picked particles by structural similarity.'''
    classifyfuncs.run_classify(
        particles=particles,
        raw_tomogram_dir=raw_tomogram_dir,
        segmentation_dir=segmentation_dir,
        output_dir=output_dir,
        voxel_size_angstrom=voxel_size_angstrom,
        box_angstrom=box_angstrom,
        n_radial_bins=n_radial_bins,
        method=method,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        n_components=n_components,
        strict_halfset_independence=strict_halfset_independence,
        random_state=random_state,
    )
