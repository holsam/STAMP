'''
STAMP: shared Typer option declarations
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated

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
