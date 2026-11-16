'''
STAMP: refine-stage plots
'''

# Import external dependencies
import matplotlib.pyplot as plt, numpy as np
from pathlib import Path

# Import internal STAMP objects
from stamp.refine.fsc import FSCResult
from stamp.utils.options.literals import PlotFormat
from stamp.utils.plotting.core import BLUE, finish, plot_path, style_axis

# fsc_curve: FSC plot with the 0.143 threshold line
def fsc_curve(ax, frequencies, correlation, resolution_angstrom):
    ax.plot(frequencies, correlation, color=BLUE, label='FSC (half A vs B, masked)')
    ax.axhline(0.143, color='k', ls='--', lw=0.8, label='0.143')
    ax.set_ylim(-0.1, 1.05)
    title = f'refinement FSC: ~{resolution_angstrom:.0f} Å' if np.isfinite(resolution_angstrom) else 'refinement FSC'
    style_axis(ax, title, xlabel='spatial frequency (1/Å)', ylabel='FSC', legend=True)

# plot_fsc: write the refinement FSC curve for one target as fsc_<target>
def plot_fsc(fsc: FSCResult, output_dir: Path, target: str, fmt: PlotFormat) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    fsc_curve(ax, fsc.frequencies_per_angstrom, fsc.fsc_masked, fsc.resolution_angstrom)
    finish(fig, plot_path(output_dir, f'fsc_{target}', fmt))
