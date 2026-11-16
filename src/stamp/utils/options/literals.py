'''
STAMP: shared Literal types for config and CLI options
'''

# Import external dependencies
from typing import Literal

# PlotFormat: image formats a plot can be saved as
PlotFormat = Literal['png', 'jpg', 'tiff', 'svg']

# PickPlotStyle: plot styles for pick and decoy position plots
PickPlotStyle = Literal['segmented', 'none']
