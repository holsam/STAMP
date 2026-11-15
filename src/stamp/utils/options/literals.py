'''
STAMP: shared Literal types for config and CLI options
'''

# Import external dependencies
from typing import Literal

# BackendName: runner backends, selectable in config and on the CLI
BackendName = Literal['local', 'mock', 'cluster']

# PlotFormat: image formats a plot can be saved as
PlotFormat = Literal['png', 'jpg', 'tiff', 'svg']
