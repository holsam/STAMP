'''
STAMP: rendering functions for filter's GUI
'''

# Import external dependencies
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pathlib import Path

# Import internal STAMP objects
from stamp.filter.gui.context import GuiContext
from stamp.filter.gui.style import BORDER_WHITE, BUTTON_BG, FG, PICK_SIZE_SCALE
from stamp.filter.render import draw_picks, draw_segmentation, load_central_slice, slice_at, volume_depth
from stamp.utils.plotting.core import bare

# visible_particles: all picks in this tomogram, or only those near the current Z plane when z_filter is on
def visible_particles(ctx: GuiContext, tomogram_id: str) -> list:
    particles = ctx.state.particles_for(tomogram_id)
    if not ctx.view['z_filter'] or ctx.view['z'] is None:
        return particles
    thickness = ctx.view['z_filter_thickness']
    return [p for p in particles if abs(p.position[2] - ctx.view['z']) <= thickness]

# draw_panel: render one background+picks panel on the given axis, returns the confidence scatter artist (or None)
def draw_panel(ctx: GuiContext, ax, path: Path | None, label: str, tomogram_id: str):
    ax.clear()
    ax.set_aspect('equal', adjustable='box')  # cla() resets this — must reapply every redraw
    ax.set_facecolor('black')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BORDER_WHITE)
        spine.set_linewidth(2.5)
    if path is None:
        ax.set_title(f'no {label} available', fontsize=9, color=FG)
        bare(ax)
        return None
    if ctx.view['z'] is None:
        plane, step = load_central_slice(path)
    else:
        plane, step = slice_at(path, ctx.view['z'])
    ctx.view['step'] = step
    if label == 'segmentation':
        draw_segmentation(ax, plane)
    else:
        ax.imshow(plane, cmap='Greys_r')
        ax.set_xlim(0, plane.shape[-1])
        ax.set_ylim(plane.shape[-2], 0)
        bare(ax)
    return draw_picks(
        ax, ctx.view['particles'], ctx.state.rejected, step,
        confidence_mode=ctx.view['confidence_mode'],
        pick_size=ctx.view['pick_size'] * PICK_SIZE_SCALE,  # scale spinbox steps by PICK_SIZE_SCALE constant
        accepted_colour=ctx.view['accepted_colour'],
        rejected_colour=ctx.view['rejected_colour'],
    )

# update_legend: show/hide the confidence colorbar to match confidence_mode
def update_legend(ctx: GuiContext) -> None:
    cbar_ax = ctx.widgets['cbar_ax']
    cbar_ax.clear()
    if ctx.view['confidence_mode']:
        cbar_ax.set_visible(True)
        cmap = LinearSegmentedColormap.from_list('confidence', [ctx.view['rejected_colour'], ctx.view['accepted_colour']])
        mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        cbar = ctx.fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal', label='confidence')
        cbar.ax.xaxis.label.set_color(FG)
        cbar.ax.tick_params(colors=FG)
    else:
        cbar_ax.set_visible(False)

# update_info_panel: refresh the left-hand tomogram/view stats
def update_info_panel(ctx: GuiContext, tomogram_id: str) -> None:
    all_particles = ctx.state.particles_for(tomogram_id)
    total = len(all_particles)
    accepted_total = sum(1 for p in all_particles if p.particle_id not in ctx.state.rejected)
    visible = ctx.view['particles']
    visible_accepted = sum(1 for p in visible if p.particle_id not in ctx.state.rejected)
    ctx.widgets['info_id_var'].set(tomogram_id)
    ctx.widgets['info_position_var'].set(f'{ctx.view["index"] + 1}/{len(ctx.state.tomogram_ids)}')
    ctx.widgets['info_status_var'].set('Reviewed' if tomogram_id in ctx.view['edited'] else 'Not yet reviewed')
    ctx.widgets['info_total_var'].set(f'Total picks: {total}')
    ctx.widgets['info_accepted_var'].set(f'Accepted: {accepted_total}')
    ctx.widgets['info_rejected_var'].set(f'Rejected: {total - accepted_total}')
    ctx.widgets['info_visible_var'].set(f'Picks visible: {len(visible)} ({visible_accepted} accepted; {len(visible) - visible_accepted} rejected)')

# draw_z_diagram: a small vertical bar showing where the current Z-filter window sits within the volume
def draw_z_diagram(ctx: GuiContext) -> None:
    canvas = ctx.widgets.get('z_diagram_canvas')
    if canvas is None:
        return
    canvas.delete('all')
    tomogram_id = ctx.current_tomogram_id()
    path = ctx.reference_path()
    depth = volume_depth(path) if path is not None else 1
    z = ctx.view['z'] if ctx.view['z'] is not None else (depth - 1) // 2
    thickness = ctx.view['z_filter_thickness']
    lo = max(0, z - thickness)
    hi = min(depth - 1, z + thickness)
    all_particles = ctx.state.particles_for(tomogram_id)
    above = sum(1 for p in all_particles if p.position[2] < lo)
    visible = sum(1 for p in all_particles if lo <= p.position[2] <= hi)
    below = sum(1 for p in all_particles if p.position[2] > hi)

    w, h = canvas.winfo_width(), canvas.winfo_height()
    if w <= 1 or h <= 1:
        return
    margin = 16  # keep Above/Below labels outside of box
    box_top, box_bottom = margin, h - margin
    box_h = box_bottom - box_top
    canvas.create_rectangle(1, box_top, w - 1, box_bottom, fill=BUTTON_BG, outline=BORDER_WHITE)
    denom = max(depth - 1, 1)
    y0 = box_top + box_h * lo / denom
    y1 = max(box_top + box_h * min(hi + 1, depth) / denom, y0 + 2)
    canvas.create_rectangle(1, y0, w - 1, y1, fill=ctx.view['accepted_colour'], outline='')
    canvas.create_text(w / 2, margin / 2, text=f'Above: {above}', fill=FG, font=('TkDefaultFont', 8))
    canvas.create_text(w / 2, min(max((y0 + y1) / 2, box_top + 10), box_bottom - 10), text=f'Visible: {visible}', fill='black', font=('TkDefaultFont', 8, 'bold'))
    canvas.create_text(w / 2, h - margin / 2, text=f'Below: {below}', fill=FG, font=('TkDefaultFont', 8))

# redraw: repaint both panels for the current tomogram/z/confidence-mode
def redraw(ctx: GuiContext) -> None:
    if not ctx.ui_ready:
        return
    tomogram_id = ctx.current_tomogram_id()
    ctx.view['visited'].add(ctx.view['index'])
    ctx.view['particles'] = visible_particles(ctx, tomogram_id)
    seg_path, raw_path = ctx.paths_for_axes()
    draw_panel(ctx, ctx.ax_seg, seg_path, 'segmentation', tomogram_id)
    draw_panel(ctx, ctx.ax_raw, raw_path, 'raw', tomogram_id)
    # reapply pan/zoom so clearing an axis doesn't reset its view to autoscale
    if ctx.view['zoom_lim'] is not None:
        xlim, ylim = ctx.view['zoom_lim']
        for ax in (ctx.ax_seg, ctx.ax_raw):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    update_legend(ctx)
    update_info_panel(ctx, tomogram_id)
    draw_z_diagram(ctx)
    ctx.fig.canvas.draw_idle()
    ctx.widgets['tomo_var'].set(tomogram_id)
    ctx.widgets['nav_position_var'].set(f'Tomogram: {ctx.view["index"] + 1}/{len(ctx.state.tomogram_ids)}')
