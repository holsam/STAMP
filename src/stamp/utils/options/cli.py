'''
STAMP: shared Typer option declarations
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated, Literal

# Import internal stamp objects
from stamp.utils.options.literals import PickPlotStyle, PlotFormat

# RawDirOption: required raw tomogram directory
RawDirOption = Annotated[
    Path,
    typer.Option('-r', '--raw-dir', exists=True, file_okay=False, help='Directory containing raw reconstructed tomograms.')
]

# OptionalRawDirOption: raw tomogram directory where it may be omitted
OptionalRawDirOption = Annotated[
    Path | None,
    typer.Option('-r', '--raw-dir', exists=True, file_okay=False, help='Directory containing raw reconstructed tomograms.')
]

# SegDirOption: segmentation directory where it may be omitted
SegDirOption = Annotated[
    Path | None,
    typer.Option('-s', '--seg-dir', exists=True, file_okay=False, help='Directory containing binary segmentations.')
]

# OutDirOption: output directory, defaults to the current directory at the command
OutDirOption = Annotated[
    Path,
    typer.Option('-o', '--out-dir', file_okay=False, help='STAMP output directory.')
]

# VoxelSizeOption: voxel size in Angstrom, read from MRC headers when omitted
VoxelSizeOption = Annotated[
    float | None,
    typer.Option('--voxel-size-a', help='Voxel size in Å. Read from MRC headers if omitted.')
]

# NProcessesOption: worker processes for a stage
NProcessesOption = Annotated[
    int,
    typer.Option('-n', '--n-processes', help='Number of processes to use (1 = sequential).')
]

# KeepRawOption: keep the raw intermediate directory instead of archiving it
KeepRawOption = Annotated[
    bool,
    typer.Option('--keep-raw', help='Keep intermediate directory instead of archiving it to a gzipped tarball.')
]

# PlotsOption: write plots
PlotsOption = Annotated[
    bool,
    typer.Option('--plots/--no-plots', help='Write plots.', rich_help_panel='Plotting')
]

# PlotFormatOption: image format for static plots
PlotFormatOption = Annotated[
    PlotFormat,
    typer.Option('--plot-format', help='Image format for static plots.', rich_help_panel='Plotting')
]

# PickPlotStyleOption: plot style for pick and decoy position plots
PickPlotStyleOption = Annotated[
    PickPlotStyle,
    typer.Option('--pick-plot-style', help='Plot style to use.', rich_help_panel='Plotting')
]

# PickZstackMovieOption: per-tomogram Z-stack movie of picks
PickZstackMovieOption = Annotated[
    bool,
    typer.Option('--pick-zstack-movie/--no-pick-zstack-movie', help='Create a per-tomogram movie stepping through Z with picks highlighted.', rich_help_panel='Plotting')
]

# PickPlot3dOption: static 3D pick visualisation per tomogram
PickPlot3dOption = Annotated[
    bool,
    typer.Option('--pick-plot-3d/--no-pick-plot-3d', help='Create a static 3D pick visualisation per tomogram.', rich_help_panel='Plotting')
]

# BoxAngstromOption: extraction box size
BoxAngstromOption = Annotated[
    float,
    typer.Option('--box-length-a', help='Extraction box edge length, in Å.', rich_help_panel = 'Subvolume extraction')
]

# NRadialBinsOption: number of radial bins in rotationsal average
NRadialBinsOption = Annotated[
    int,
    typer.Option('--n-bins', help='Radial bins in the rotational average.', rich_help_panel = 'Subvolume extraction'),
]

# MethodOption: clustering method
MethodOption = Annotated[
    Literal['leiden', 'hdbscan', 'kmeans'],
    typer.Option('--method', help='Clustering method to use.', rich_help_panel = 'Clustering'),
]

# MinClusterSizeOption: minimum cluster size
MinClusterSizeOption = Annotated[
    int,
    typer.Option('--min-cluster-size', help='Minimum cluster size (used by HDBSCAN and Leiden, and the smallest seed the polish step refines).', rich_help_panel = 'Clustering'),
]

# NClustersOption: number of KMeans clusters
NClustersOption = Annotated[
    int,
    typer.Option('--n-clusters', help='Number of clusters (used by KMeans).', rich_help_panel = 'Clustering'),
]

# NComponentsOption: number of PCA components to use
NComponentsOption = Annotated[
    int,
    typer.Option('--n-components', help='Number of PCA components.', rich_help_panel = 'Clustering'),
]

# StrictHalfsetIndependenceOption: whether to use strict halfset independence when clustering
StrictHalfsetIndependenceOption = Annotated[
    bool,
    typer.Option('--strict-halfset-independence/--no-strict-halfset-independence', help='Cluster each half separately and match clusters afterwards.', rich_help_panel = 'Clustering'),
]

# RandomStateOption: seed to use for PCA/KMeans
RandomStateOption = Annotated[
    int,
    typer.Option('--seed', help='Seed for PCA and KMeans.', rich_help_panel = 'Clustering'),
]

# InplaneAngularStepDegreesOption: angular steps to use for azimuth search
InplaneAngularStepDegreesOption = Annotated[
    float, typer.Option('--inplane-step-deg', help='Angular step of the azimuth search in degrees.', rich_help_panel = 'In-plane alignment'),
]

# AzimuthalModesOption: highest azimuthal Fourier mode to keep
AzimuthalModesOption = Annotated[
    int,
    typer.Option(help='Highest azimuthal Fourier mode retained. 0 reproduces pure rotational averaging; 4 captures C4 symmetry. Higher modes are increasingly noisy.', rich_help_panel = 'Subvolume extraction'),
]

# MinRadiusFractionOption: radial bins to exclude
MinRadiusFractionOption = Annotated[
    float,
    typer.Option(help='Radial bins below this fraction of the box radius are excluded from azimuthal features (too few voxels to report symmetry).', rich_help_panel = 'Subvolume extraction'),
]

# NAzimuthalSamplesOption: number of azimuthal sampling points
NAzimuthalSamplesOption = Annotated[
    int,
    typer.Option(help='Azimuthal sampling points (must be at least 2*(modes+1)).', rich_help_panel = 'Subvolume extraction'),
]

# FeatureNormalisationOption: method to normalise azimuthal modes
FeatureNormalisationOption = Annotated[
    Literal['none', 'particle', 'tomogram', 'both'],
    typer.Option('--feature-normalisation', help='Method for azimuthal mode (above 0) normalisation.', rich_help_panel = 'Subvolume extraction'),
]

# TiltMaxDegreesOption: maximum tilt of tilt series
TiltMaxDegreesOption = Annotated[
    float | None,
    typer.Option('--tilt-max', help='Maximum tilt of the tilt series in degrees, sets the missing wedge. Omit for no wedge.', min=1.0, max=89.0, rich_help_panel = 'Polish'),
]

# TiltAxisDegreesOption: axis of tilt series
TiltAxisDegreesOption = Annotated[
    float,
    typer.Option('--tilt-axis', help='Tilt axis angle from the tomogram x axis in the xy plane, 90 is a tilt about y.', rich_help_panel = 'Polish'),
]

# BinFactorOption: binning factor to use for polishing
BinFactorOption = Annotated[
    int,
    typer.Option('--bin-factor', help='Fourier binning factor of the polish working resolution.', min=1, rich_help_panel = 'Polish'),
]

# NRotationsOption: candidate rotations of none prior
NRotationsOption = Annotated[
    int,
    typer.Option('--n-rots', help='Candidate rotations of the none prior search.', min=2, rich_help_panel = 'Polish'),
]

# ShiftRangeVoxelsOption: largest shift search to use
ShiftRangeVoxelsOption = Annotated[
    int,
    typer.Option('--shift-range', help='Largest shift searched (the axis prior also searches a quarter box along the axis).', min=0, rich_help_panel = 'Polish'),
]

# PolishRoundsOption: number of polishing rounds
PolishRoundsOption = Annotated[
    int,
    typer.Option('--polish-rounds', help='Number of reassign, rebuild and merge rounds during polish.', min=1, rich_help_panel = 'Polish'),
]

# NullSigmasOption: null standard deviations to beat phase-randomised by
NullSigmasOption = Annotated[
    float,
    typer.Option('--null-sigmas', help='Number of null standard deviations that a reference must beat the phase-randomised null agreement by.', min=0.0, rich_help_panel = 'Polish'),
]

# RejectQuantileOption: null quantile of best correlation to accept signal as
RejectQuantileOption = Annotated[
    float,
    typer.Option('--reject-quantile', help='Null quantile of the best correlation below which a particle is rejected as noise.', min=0.5, max=0.999, rich_help_panel = 'Polish'),
]

# MergeFractionOption: fraction of agreement to reach for merging
MergeFractionOption = Annotated[
    float,
    typer.Option('--merge-fraction', help='Fraction of the higher half-split agreement two references must reach to be merged (and two halves to be matched).', min=0.0, rich_help_panel = 'Polish'),
]
