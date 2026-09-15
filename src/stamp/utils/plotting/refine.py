'''
STAMP: refine-stage plots
'''

# Import external dependencies
import numpy as np

# Import internal STAMP objects
from stamp.utils.plotting.core import BLUE, style_axis

# fsc_curve: FSC plot with the 0.143 threshold line
def fsc_curve(ax, frequencies, correlation, resolution_angstrom):
    ax.plot(frequencies, correlation, color=BLUE, label='FSC (half A vs B, masked)')
    ax.axhline(0.143, color='k', ls='--', lw=0.8, label='0.143')
    ax.set_ylim(-0.1, 1.05)
    title = f'refinement FSC: ~{resolution_angstrom:.0f} Å' if np.isfinite(resolution_angstrom) else 'refinement FSC'
    style_axis(ax, title, xlabel='spatial frequency (1/Å)', ylabel='FSC', legend=True)
