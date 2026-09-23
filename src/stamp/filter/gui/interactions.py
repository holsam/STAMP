'''
STAMP: event handlers and view-math for filter's GUI
'''

# Import external dependencies
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath

# Import internal STAMP objects
from stamp.filter.gui.context import GuiContext
from stamp.filter.gui.rendering import redraw
from stamp.filter.gui.style import BORDER_WHITE
from stamp.filter.render import load_central_slice, slice_at

# on_pick: toggle the clicked particle
def on_pick(ctx: GuiContext, event) -> None:
    if not len(event.ind) or not hasattr(event.artist, 'particle_ids'):
        return
    from stamp.filter.gui import actions
    particle_id = event.artist.particle_ids[event.ind[0]]
    actions.snapshot(ctx)
    ctx.state.toggle(particle_id)
    ctx.view['edited'].add(ctx.current_tomogram_id())
    redraw(ctx)
    actions.autosave(ctx)

# set_button_tool_state: style-highlight a toggle button when active
def set_button_tool_state(ctx: GuiContext, key: str, icon_name: str, active: bool) -> None:
    from stamp.filter.gui.components import icon
    button = ctx.widgets[key]
    button.configure(style='ToolActive.TButton' if active else 'Dark.TButton')
    variant = f'{icon_name}_pressed' if active else icon_name
    image = icon(variant)
    if image is not None:
        ctx.widgets[f'_icon_{variant}'] = image  # kept referenced to avoid garbage collection
        button.configure(image=image)

# set_tool: set current tool
def set_tool(ctx: GuiContext, tool: str) -> None:
    ctx.view['tool'] = 'pan' if ctx.view['tool'] == tool else tool
    lasso_active = ctx.view['tool'] in ('lasso_reject', 'lasso_accept')
    ctx.widgets['lasso_seg'].set_active(lasso_active)
    ctx.widgets['lasso_raw'].set_active(lasso_active)
    set_button_tool_state(ctx, 'lasso_reject_btn', 'lasso_reject', ctx.view['tool'] == 'lasso_reject')
    set_button_tool_state(ctx, 'lasso_accept_btn', 'lasso_accept', ctx.view['tool'] == 'lasso_accept')
    set_button_tool_state(ctx, 'zoom_rect_btn', 'zoom_rect', ctx.view['tool'] == 'zoom_rect')

# on_lasso_select: apply the active lasso tool to every displayed particle enclosed by the path
def on_lasso_select(ctx: GuiContext, vertices) -> None:
    if ctx.view['tool'] not in ('lasso_reject', 'lasso_accept') or not ctx.view['particles']:
        ctx.fig.canvas.draw_idle()  # clear drawn line if no particles selected
        return
    enclosed = MplPath(vertices)
    xy = [(p.position[0] / ctx.view['step'], p.position[1] / ctx.view['step']) for p in ctx.view['particles']]
    inside = enclosed.contains_points(xy)
    ids = {p.particle_id for p, is_inside in zip(ctx.view['particles'], inside) if is_inside}
    if not ids:
        ctx.fig.canvas.draw_idle()
        return
    from stamp.filter.gui import actions
    actions.snapshot(ctx)
    if ctx.view['tool'] == 'lasso_reject':
        ctx.state.rejected.update(ids)
    else:
        ctx.state.rejected.difference_update(ids)
    ctx.view['edited'].add(ctx.current_tomogram_id())
    redraw(ctx)
    actions.autosave(ctx)

# set_view: apply x/y limits to both panels and remember these so the next redraw can restore them
def set_view(ctx: GuiContext, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    for ax in (ctx.ax_seg, ctx.ax_raw):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ctx.view['zoom_lim'] = (xlim, ylim)
    ctx.fig.canvas.draw_idle()

# on_pan_press: activate panning tool
def on_pan_press(ctx: GuiContext, event) -> None:
    if ctx.view['tool'] != 'pan' or event.inaxes not in (ctx.ax_seg, ctx.ax_raw) or event.button != 1:
        return
    ctx.pan_state.update(active=True, x0=event.xdata, y0=event.ydata, xlim0=event.inaxes.get_xlim(), ylim0=event.inaxes.get_ylim())

# on_pan_motion: move viewer with mouse
def on_pan_motion(ctx: GuiContext, event) -> None:
    if not ctx.pan_state['active'] or event.xdata is None or event.ydata is None:
        return
    dx = event.xdata - ctx.pan_state['x0']
    dy = event.ydata - ctx.pan_state['y0']
    x0, x1 = ctx.pan_state['xlim0']
    y0, y1 = ctx.pan_state['ylim0']
    set_view(ctx, (x0 - dx, x1 - dx), (y0 - dy, y1 - dy))

# on_pan_release: stop moving viewer with mouse
def on_pan_release(ctx: GuiContext, _event) -> None:
    ctx.pan_state['active'] = False

# on_zoomrect_press: activate rectangle patch to draw zoom region
def on_zoomrect_press(ctx: GuiContext, event) -> None:
    if ctx.view['tool'] != 'zoom_rect' or event.inaxes not in (ctx.ax_seg, ctx.ax_raw) or event.button != 1:
        return
    patch = Rectangle((event.xdata, event.ydata), 0, 0, fill=False, edgecolor=BORDER_WHITE, linewidth=1.5, linestyle='--')
    event.inaxes.add_patch(patch)
    ctx.zoomrect_state.update(active=True, ax=event.inaxes, x0=event.xdata, y0=event.ydata, patch=patch)

# on_zoomrect_motion: draw zoom rectangle
def on_zoomrect_motion(ctx: GuiContext, event) -> None:
    if not ctx.zoomrect_state['active'] or event.xdata is None or event.ydata is None:
        return
    x0, y0 = ctx.zoomrect_state['x0'], ctx.zoomrect_state['y0']
    ctx.zoomrect_state['patch'].set_bounds(min(x0, event.xdata), min(y0, event.ydata), abs(event.xdata - x0), abs(event.ydata - y0))
    ctx.fig.canvas.draw_idle()

# on_zoomrect_release: zoom to drawn region
def on_zoomrect_release(ctx: GuiContext, event) -> None:
    if not ctx.zoomrect_state['active']:
        return
    ctx.zoomrect_state['active'] = False
    ax = ctx.zoomrect_state['ax']
    patch = ctx.zoomrect_state['patch']
    ctx.zoomrect_state['patch'] = None
    if patch is not None:
        patch.remove()
    x0, y0 = ctx.zoomrect_state['x0'], ctx.zoomrect_state['y0']
    if ax is None or event.xdata is None or event.ydata is None or x0 == event.xdata or y0 == event.ydata:
        ctx.fig.canvas.draw_idle()
        return
    cur_y0, cur_y1 = ax.get_ylim()
    xlim = tuple(sorted((x0, event.xdata)))
    y_sorted = sorted((y0, event.ydata))
    ylim = (y_sorted[1], y_sorted[0]) if cur_y0 > cur_y1 else tuple(y_sorted)
    set_view(ctx, xlim, ylim)
    set_tool(ctx, 'pan')

# zoom: scale both panels' view around their current centre; factor < 1 zooms in
def zoom(ctx: GuiContext, factor: float) -> None:
    x0, x1 = ctx.ax_seg.get_xlim()
    y0, y1 = ctx.ax_seg.get_ylim()
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    set_view(ctx, (xc - (xc - x0) * factor, xc + (x1 - xc) * factor), (yc - (yc - y0) * factor, yc + (y1 - yc) * factor))

# fit_view_limits: a square field of view (side = max(h, w)) centred on the image
def fit_view_limits(ctx: GuiContext):
    path = ctx.reference_path()
    if path is None:
        return None
    plane, _step = load_central_slice(path) if ctx.view['z'] is None else slice_at(path, ctx.view['z'])
    h, w = plane.shape[-2], plane.shape[-1]
    side = max(h, w)
    cx, cy = w / 2, h / 2
    return (cx - side / 2, cx + side / 2), (cy + side / 2, cy - side / 2)

# zoom_fit: whole image visible, letterboxed (padded) within a square view if not itself square
def zoom_fit(ctx: GuiContext) -> None:
    limits = fit_view_limits(ctx)
    if limits is None:
        ctx.view['zoom_lim'] = None
        redraw(ctx)
        return
    set_view(ctx, *limits)

# zoom_fill: crop to the shorter dimension so the image fills the square view with no letterboxing
def zoom_fill(ctx: GuiContext) -> None:
    path = ctx.reference_path()
    if path is None:
        return
    plane, _step = load_central_slice(path) if ctx.view['z'] is None else slice_at(path, ctx.view['z'])
    h, w = plane.shape[-2], plane.shape[-1]
    side = min(h, w)
    cx, cy = w / 2, h / 2
    set_view(ctx, (cx - side / 2, cx + side / 2), (cy + side / 2, cy - side / 2))

# on_key: arrow-key tomogram navigation, ctrl/cmd+z undo, ctrl/cmd +/- zoom
def on_key(ctx: GuiContext, event) -> None:
    from stamp.filter.gui import actions
    from stamp.filter.gui.style import REDO_KEYS, UNDO_KEYS, ZOOM_IN_KEYS, ZOOM_OUT_KEYS
    if event.key == 'right':
        actions.step(ctx, 1)()
    elif event.key == 'left':
        actions.step(ctx, -1)()
    elif event.key in REDO_KEYS:
        actions.redo(ctx)
    elif event.key in UNDO_KEYS:
        actions.undo(ctx)
    elif event.key in ZOOM_IN_KEYS:
        zoom(ctx, 0.8)
    elif event.key in ZOOM_OUT_KEYS:
        zoom(ctx, 1.25)

# force_release: ensure mouse releases over Tk widgets still register
def force_release(ctx: GuiContext, _event=None) -> None:
    ctx.pan_state['active'] = False
    if ctx.zoomrect_state['active']:
        ctx.zoomrect_state['active'] = False
        if ctx.zoomrect_state['patch'] is not None:
            ctx.zoomrect_state['patch'].remove()
            ctx.zoomrect_state['patch'] = None
            ctx.fig.canvas.draw_idle()
