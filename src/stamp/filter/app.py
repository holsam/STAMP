'''
STAMP: GUI entry point for filter command
'''

# Import external dependencies
import matplotlib, matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from pathlib import Path

# Import internal STAMP objects
from stamp.filter.gui import actions, interactions, layout
from stamp.filter.gui.context import GuiContext
from stamp.filter.gui.rendering import redraw
from stamp.filter.gui.style import (
    BORDER_WHITE, BG,
    DEFAULT_ACCEPTED_COLOUR, DEFAULT_PICK_SIZE, DEFAULT_REJECTED_COLOUR, DEFAULT_Z_FILTER_THICKNESS,
)
from stamp.filter.state import FilterState
from stamp.utils.log import log

# Use matplotlib's TkAgg backend
matplotlib.use('TkAgg', force=True)
# Disable default matplotlib toolbar
matplotlib.rcParams['toolbar'] = 'none'

# run_filter_gui: launch filter's GUI
def run_filter_gui(state: FilterState, output_dir: Path) -> None:
    fig, (ax_seg, ax_raw) = plt.subplots(1, 2, figsize=(10, 6.5))
    fig.patch.set_facecolor(BG)
    for ax in (ax_seg, ax_raw):
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BORDER_WHITE)
            spine.set_linewidth(2.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.08, wspace=0.04)

    ctx = GuiContext(
        state=state,
        output_dir=output_dir,
        fig=fig,
        ax_seg=ax_seg,
        ax_raw=ax_raw,
        root=fig.canvas.manager.window,
    )
    ctx.view = {
        'index': 0,
        'step': 1,
        'particles': [],
        'z': None,
        'confidence_mode': False,
        'z_filter': True,
        'z_filter_thickness': DEFAULT_Z_FILTER_THICKNESS,
        'tool': 'pan',  # 'pan' | 'lasso_reject' | 'lasso_accept' | 'zoom_rect'
        'pick_size': DEFAULT_PICK_SIZE,
        'accepted_colour': DEFAULT_ACCEPTED_COLOUR,
        'rejected_colour': DEFAULT_REJECTED_COLOUR,
        'visited': set(),
        'edited': set(),  # tomogram_ids with at least one explicit accept/reject/lasso/accept-all/reject-all edit
        'zoom_lim': None,  # (xlim, ylim) to keep across a redraw; None = autoscale/full view
    }
    ctx.pan_state = {'active': False, 'x0': None, 'y0': None, 'xlim0': None, 'ylim0': None}
    ctx.zoomrect_state = {'active': False, 'ax': None, 'x0': None, 'y0': None, 'patch': None}

    layout.build(ctx)

    fig.canvas.mpl_connect('pick_event', lambda event: interactions.on_pick(ctx, event))
    fig.canvas.mpl_connect('key_press_event', lambda event: interactions.on_key(ctx, event))
    fig.canvas.mpl_connect('button_press_event', lambda event: interactions.on_pan_press(ctx, event))
    fig.canvas.mpl_connect('motion_notify_event', lambda event: interactions.on_pan_motion(ctx, event))
    fig.canvas.mpl_connect('button_release_event', lambda event: interactions.on_pan_release(ctx, event))
    fig.canvas.mpl_connect('button_press_event', lambda event: interactions.on_zoomrect_press(ctx, event))
    fig.canvas.mpl_connect('motion_notify_event', lambda event: interactions.on_zoomrect_motion(ctx, event))
    fig.canvas.mpl_connect('button_release_event', lambda event: interactions.on_zoomrect_release(ctx, event))

    # Enforce a safety net so drags ending over widget still reach Tk
    ctx.root.bind_all('<ButtonRelease-1>', lambda event=None: interactions.force_release(ctx, event), add='+')

    # Keep LassoSelectors referenced to avoid garbage collection
    ctx.widgets['lasso_seg'] = LassoSelector(ax_seg, onselect=lambda vertices: interactions.on_lasso_select(ctx, vertices))
    ctx.widgets['lasso_raw'] = LassoSelector(ax_raw, onselect=lambda vertices: interactions.on_lasso_select(ctx, vertices))
    ctx.widgets['lasso_seg'].set_active(False)
    ctx.widgets['lasso_raw'].set_active(False)

    # ui_ready gates redraw until every widget referenced inside it has been built
    ctx.ui_ready = True
    redraw(ctx)
    ctx.root.after(150, lambda: interactions.zoom_fit(ctx))
    log.info('Click a pick to toggle. Drag to pan viewer windows. Pick a lasso or zoom-to-rectangle tool to change that, click it again to return to pan. Arrow keys navigate, ctrl/cmd+z undo, ctrl/cmd +/- zoom. Autosaves after every change.')
    plt.show()

    actions.autosave(ctx)
    log.info(f'Filtered particle set saved to {output_dir}')
