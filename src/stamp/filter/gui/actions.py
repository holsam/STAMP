'''
STAMP: commands for editing filter's GUI GuiContext
'''

# Import external dependencies
from tkinter.colorchooser import askcolor

# Import internal STAMP objects
from stamp.filter.gui.context import GuiContext
from stamp.filter.gui.interactions import fit_view_limits
from stamp.filter.gui.rendering import redraw
from stamp.utils.log import log

# autosave: persist state.rejected after changes
def autosave(ctx: GuiContext) -> None:
    try:
        ctx.state.save(ctx.output_dir)
    except Exception as error:  # noqa: BLE001 - disk full etc, log and keep going
        log.warning(f'Autosave failed: {error!r}')

# snapshot: push current rejected state before a change and clear redo stack
def snapshot(ctx: GuiContext) -> None:
    ctx.history.append(frozenset(ctx.state.rejected))
    ctx.future.clear()

# undo: restore the previous rejected-set snapshot
def undo(ctx: GuiContext) -> None:
    if not ctx.history:
        return
    ctx.future.append(frozenset(ctx.state.rejected))
    ctx.state.rejected = set(ctx.history.pop())
    redraw(ctx)
    autosave(ctx)

# redo: restore the next rejected-set snapshot
def redo(ctx: GuiContext) -> None:
    if not ctx.future:
        return
    ctx.history.append(frozenset(ctx.state.rejected))
    ctx.state.rejected = set(ctx.future.pop())
    redraw(ctx)
    autosave(ctx)

# reject_all: mark all particles in current tomogram
def reject_all(ctx: GuiContext) -> None:
    snapshot(ctx)
    ctx.state.reject_all(ctx.current_tomogram_id())
    ctx.view['edited'].add(ctx.current_tomogram_id())
    redraw(ctx)
    autosave(ctx)

# accept_all: clear all particles in current tomogram
def accept_all(ctx: GuiContext) -> None:
    snapshot(ctx)
    ctx.state.reset_tomogram(ctx.current_tomogram_id())
    ctx.view['edited'].add(ctx.current_tomogram_id())
    redraw(ctx)
    autosave(ctx)

# reset_tomogram: clear rejections & reviewed flag
def reset_tomogram(ctx: GuiContext) -> None:
    snapshot(ctx)
    ctx.state.reset_tomogram(ctx.current_tomogram_id())
    ctx.view['edited'].discard(ctx.current_tomogram_id())
    redraw(ctx)
    autosave(ctx)

# set_confidence_mode: toolbar control for confidence rendering
def set_confidence_mode(ctx: GuiContext, value: bool) -> None:
    ctx.view['confidence_mode'] = value
    redraw(ctx)

# set_z_filter: toolbar control for Z-filtering picks
def set_z_filter(ctx: GuiContext, value: bool) -> None:
    ctx.view['z_filter'] = value
    redraw(ctx)

# set_z_thickness: toolbar control for Z slices to show as visible
def set_z_thickness(ctx: GuiContext, value: int) -> None:
    ctx.view['z_filter_thickness'] = value
    redraw(ctx)

# set_pick_size: toolbar control for setting pick size
def set_pick_size(ctx: GuiContext, value: int) -> None:
    ctx.view['pick_size'] = value
    redraw(ctx)

# pick_colour: open the native colour chooser for accepted/rejected markers & update the swatch
def pick_colour(ctx: GuiContext, which: str) -> None:
    key = f'{which}_colour'
    _, hex_colour = askcolor(color=ctx.view[key], title=f'{which.title()} pick colour')
    if hex_colour is None:
        return
    ctx.view[key] = hex_colour
    ctx.widgets[f'{which}_swatch'].itemconfig(ctx.widgets[f'{which}_swatch_id'], fill=hex_colour)
    redraw(ctx)

# on_zslider: redraw at the chosen Z plane
def on_zslider(ctx: GuiContext, z: str) -> None:
    ctx.view['z'] = int(float(z))
    redraw(ctx)

# go_to_tomogram: resets Z to centre and the view to fullzoom then redraws
def go_to_tomogram(ctx: GuiContext) -> None:
    from stamp.filter.render import volume_depth
    path = ctx.reference_path()
    if path is not None:
        center = (volume_depth(path) - 1) // 2
        ctx.view['z'] = center
        ctx.widgets['z_scale'].set(center)
    else:
        ctx.view['z'] = None
    ctx.view['zoom_lim'] = fit_view_limits(ctx)  # square field of view, not just "whatever imshow defaults to"
    redraw(ctx)

# step: move to the next/previous tomogram
def step(ctx: GuiContext, delta: int):
    def _handler() -> None:
        ctx.view['index'] = (ctx.view['index'] + delta) % len(ctx.state.tomogram_ids)
        go_to_tomogram(ctx)
    return _handler

# jump_to_tomogram: switch to the tomogram chosen in the dropdown
def jump_to_tomogram(ctx: GuiContext, tomogram_id: str) -> None:
    if tomogram_id not in ctx.state.tomogram_ids:
        return
    ctx.view['index'] = ctx.state.tomogram_ids.index(tomogram_id)
    go_to_tomogram(ctx)
