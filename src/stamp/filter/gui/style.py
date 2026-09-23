'''
STAMP: style constants for filter's GUI
'''

# Import external dependencies
from pathlib import Path

# MAX_ICON_PX: maximum size for button icons regardless of source PNG resolution
MAX_ICON_PX = 20

# DEFAULT_PICK_SIZE: default size to use for each pick
DEFAULT_PICK_SIZE = 10

# DEFAULT_ACCEPTED_COLOUR: default colour to use for accepted picks (green)
DEFAULT_ACCEPTED_COLOUR = '#007e5d'

# DEFAULT_REJECTED_COLOUR default colour to use for rejected picks (red)
DEFAULT_REJECTED_COLOUR = '#ff5d5d'

# DEFAULT_Z_FILTER_THICKNESS: default number of Z slices to show around current Z plane
DEFAULT_Z_FILTER_THICKNESS = 5

# PICK_SIZE_SCALE: scale factor for increasing pick size (default steps are imperceptible)
PICK_SIZE_SCALE = 10

# UNDO_KEYS: keybindings for undo button
UNDO_KEYS = {'ctrl+z', 'cmd+z', 'super+z'}

# REDO_KEYS: keybindings for redo button
REDO_KEYS = {'ctrl+shift+z', 'ctrl+Z', 'cmd+shift+z', 'cmd+Z', 'super+shift+z', 'super+Z'}

# ZOOM_IN_KEYS: keybindings for zoom in button
ZOOM_IN_KEYS = {'ctrl++', 'ctrl+=', 'cmd++', 'cmd+=', 'super++', 'super+='}

# ZOOM_OUT_KEYS: keybindings for zoom out button
ZOOM_OUT_KEYS = {'ctrl+-', 'cmd+-', 'super+-'}

# BG: dark theme window background colour
BG = '#1e1e1e'

# PANEL_BG: dark theme panel background colour
PANEL_BG = '#252526'

# FG: dark theme foreground element
FG = '#e6e6e6'

# BUTTON_BG: dark theme background colour for buttons
BUTTON_BG = PANEL_BG

# BUTTON_ACTIVE_BG: dark theme button background for active tools
BUTTON_ACTIVE_BG = '#4a4d4f'

# BORDER_WHITE: dark theme white border for viewer boxes
BORDER_WHITE = '#f0f0f0'

# ACCENT: dark theme highlight for active tools
ACCENT = '#5a5d5f'

# ICON_DIR: directory containing button icons
ICON_DIR = Path(__file__).parent / 'icons'
