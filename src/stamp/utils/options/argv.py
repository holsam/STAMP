'''
STAMP: shared argv assembly for stage ToolCommands
'''

# Import external dependencies
from pathlib import Path

# Import internal stamp objects
from stamp.schemas.config import PlotSettings

# plot_argv: plot flags for a stage's ToolCommand; pick_options adds the pick/decoy-only style and 3D/movie flags
def plot_argv(plots: PlotSettings, *, pick_options: bool = False) -> list[str]:
    if pick_options:
        # pick/decoy default tp no plots, so style/format flags are only sent if plotting
        argv = ['--plots', '--pick-plot-style', plots.pick_style, '--plot-format', plots.format] if plots.enabled else []
        argv.append('--pick-zstack-movie' if plots.pick_zstack_movie else '--no-pick-zstack-movie')
        argv.append('--pick-plot-3d' if plots.pick_plot_3d else '--no-pick-plot-3d')
        return argv
    return ['--plots' if plots.enabled else '--no-plots', '--plot-format', plots.format]

# seg_dir_argv: --seg-dir for a segmentation directory, nothing when none is configured
def seg_dir_argv(segmentation_dir: Path | None) -> list[str]:
    return [] if segmentation_dir is None else ['--seg-dir', str(segmentation_dir)]

# keep_raw_argv: --keep-raw only when kept; the CLI flags have no --no-keep-raw form, so omission means archive
def keep_raw_argv(keep_raw: bool) -> list[str]:
    return ['--keep-raw'] if keep_raw else []
