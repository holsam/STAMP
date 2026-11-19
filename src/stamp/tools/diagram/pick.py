'''
STAMP: detailed pick workflow diagram
'''

# Import external dependencies
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.render import DiagramRun, render_diagram
from stamp.tools.diagram.utils.plotting import draw_carriers
from stamp.utils.plotting.core import BLUE, GREEN, GREY, ORANGE, RED, finish, frame, show_segmentation

# PickRun: only the scene and the pick stage are needed for the pick walkthrough (DiagramRun default)
class PickRun(DiagramRun):
    pass

# --- panels ---
# plot_raw: raw tomogram slice
def plot_raw(run, ax):
    w = run.w
    ax.imshow(w.raw_tomogram, cmap=GREY)
    frame(ax, w.field_px, 'raw tomogram slice')

# plot_segmentation: carrier segmentation (grey) with protein density (orange); only the normal prior reads a segmentation
def plot_segmentation(run, ax):
    w = run.w
    show_segmentation(ax, w.segmentation, w.field_px)
    frame(ax, w.field_px, 'membrane segmentation (grey) + protein density (orange)')

# plot_files: two picks files as an external picker would write them
def plot_files(run, ax):
    w = run.w
    draw_carriers(ax, w.carriers, colour='0.7', lw=0.6)
    ax.scatter(w.picks_a[:, 0], w.picks_a[:, 1], s=55, facecolors='none', edgecolors=BLUE, linewidths=1.1, label=f'picks file A (n={len(w.picks_a)})')
    ax.scatter(w.picks_b[:, 0], w.picks_b[:, 1], s=95, facecolors='none', edgecolors=ORANGE, linewidths=1.1, label=f'picks file B (n={len(w.picks_b)})')
    frame(ax, w.field_px, 'import: one picks file per picker (CSV, STAR, JSON)')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.4)

# plot_orientation: orientation arrows per pick, picks with no orientation marked
def plot_orientation(run, ax):
    w = run.w
    draw_carriers(ax, w.carriers, colour='0.7', lw=0.6)
    points = np.vstack([w.picks_a, w.picks_b])
    angle = np.concatenate([w.angle_a, w.angle_b])
    oriented = np.concatenate([w.oriented_a, w.oriented_b])
    if w.prior is not OrientationPrior.NONE:
        ax.quiver(points[oriented, 0], points[oriented, 1], np.cos(angle[oriented]), np.sin(angle[oriented]), angles='xy', scale_units='xy', scale=1 / (0.45 * w.box), width=0.003, color=GREEN)
    if (~oriented).any():
        ax.scatter(points[~oriented, 0], points[~oriented, 1], s=22, c=RED, marker='x', label=f'no orientation (n={int((~oriented).sum())})')
        ax.legend(fontsize=6, loc='upper right', framealpha=0.4)
    how = {OrientationPrior.NORMAL: 'normal inferred from the nearest membrane surface', OrientationPrior.AXIS: 'filament direction read from the picks file', OrientationPrior.NONE: 'orientations ignored, none is used'}[w.prior]
    frame(ax, w.field_px, f'orientation: {how}')

# plot_consensus: points both files agree on, plus one-file-only rejects
def plot_consensus(run, ax):
    w = run.w
    draw_carriers(ax, w.carriers, colour='0.7', lw=0.6)
    if len(w.rejected_picks):
        ax.scatter(w.rejected_picks[:, 0], w.rejected_picks[:, 1], s=40, c=RED, marker='x', label=f'one file only (n={len(w.rejected_picks)})')
    if len(w.reconciled):
        ax.scatter(w.reconciled[:, 0], w.reconciled[:, 1], s=26, c=GREEN, label=f'consensus (n={len(w.reconciled)})')
    frame(ax, w.field_px, 'reconcile: keep points both files agree on (intersection)')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.4)

# plot_filters: consensus picks dropped for having no orientation or a steep beam angle (normal prior)
def plot_filters(run, ax):
    w = run.w
    draw_carriers(ax, w.carriers, colour='0.7', lw=0.6)
    unoriented = ~w.reconciled_oriented
    for mask, colour, marker, label in ((w.beam_dropped, ORANGE, 'x', f'beam angle > {run.config.max_beam_angle_deviation:g} deg'), (unoriented, RED, 'x', 'no orientation')):
        if mask.any():
            ax.scatter(w.reconciled[mask, 0], w.reconciled[mask, 1], s=40, c=colour, marker=marker, label=f'dropped: {label} (n={int(mask.sum())})')
    ax.scatter(w.consensus[:, 0], w.consensus[:, 1], s=24, c=GREEN, label=f'kept (n={len(w.consensus)})')
    frame(ax, w.field_px, 'filters: drop unoriented and steeply tilted picks')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.4)

# plot_halfsets: half-set split (whole carrier to one half) with decoys
def plot_halfsets(run, ax):
    w = run.w
    draw_carriers(ax, w.carriers, colour='0.7', lw=0.6)
    for half, colour in ((0, BLUE), (1, ORANGE)):
        sel = w.consensus_half == half
        if sel.any():
            ax.scatter(w.consensus[sel, 0], w.consensus[sel, 1], s=24, c=colour, label=f'half {half + 1} (n={int(sel.sum())})')
    if len(w.decoy_points):
        ax.scatter(w.decoy_points[:, 0], w.decoy_points[:, 1], s=30, c='0.5', marker='x', label=f'decoys (n={len(w.decoy_points)})')
    frame(ax, w.field_px, 'half-set split (whole carrier -> one half)' if w.carriers else 'half-set split (per particle)')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.4)

# plot_final: the final consensus particle set on the tomogram
def plot_final(run, ax):
    w = run.w
    ax.imshow(w.raw_tomogram, cmap=GREY)
    if len(w.consensus):
        ax.scatter(w.consensus[:, 0], w.consensus[:, 1], s=18, c=GREEN)
    frame(ax, w.field_px, f'final particle set on the tomogram (n={len(w.consensus)})')

# --- drivers ---
# _panels: ordered (filename stem, plot function) pairs for the pick walkthrough; the segmentation and filters only exist for the normal prior
def _panels(run) -> list:
    normal = run.w.prior is OrientationPrior.NORMAL
    return [('raw_tomogram', plot_raw)] + ([('segmentation', plot_segmentation)] if normal else []) + [('picks_files', plot_files), ('orientation', plot_orientation), ('consensus', plot_consensus)] + ([('filters', plot_filters)] if normal else []) + [('halfsets', plot_halfsets), ('final_particles', plot_final)]

# save_panels: write each panel as its own numbered PNG
def save_panels(run, outdir):
    for number, (name, fn) in enumerate(_panels(run), start=1):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fn(run, ax)
        finish(fig, outdir / f'{number:02d}_{name}.png')

# save_storyboard: lay every pick panel out on one 00_pick_storyboard.png figure
def save_storyboard(run, outdir):
    panels = _panels(run)
    fig = plt.figure(figsize=(21, 12))
    grid = fig.add_gridspec(2, 4, hspace=0.2, wspace=0.2)
    for index, (_name, fn) in enumerate(panels):
        fn(run, fig.add_subplot(grid[index // 4, index % 4]))
    finish(fig, outdir / '00_pick_storyboard.png')

# render: run the mocked picking stage and write its figures, returning the output directory
def render(config: DiagramConfig) -> Path:
    return render_diagram(config, PickRun, save_panels, save_storyboard)
