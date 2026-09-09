'''
STAMP: detailed pick workflow diagram
'''

# Import external dependencies
import matplotlib.pyplot as plt
from pathlib import Path

# Import internal STAMP objects
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.render import DiagramRun, render_diagram
from stamp.tools.diagram.utils.plotting import BLUE, GREEN, GREY, ORANGE, RED, draw_vesicles, finish, frame, show_segmentation

# PickRun: only the scene is needed for the pick walkthrough (DiagramRun default)
class PickRun(DiagramRun):
    pass

# --- panels ---
# plot_raw: raw tomogram slice
def plot_raw(run, ax):
    w = run.w
    ax.imshow(w.raw_tomogram, cmap=GREY)
    frame(ax, w.field_px, 'raw tomogram slice')

# plot_segmentation: membrane segmentation (grey) with protein density (orange)
def plot_segmentation(run, ax):
    w = run.w
    show_segmentation(ax, w.segmentation, w.field_px)
    frame(ax, w.field_px, 'membrane segmentation (grey) + protein density (orange)')

# plot_surface_score: surface points coloured by score along each membrane
def plot_surface_score(run, ax):
    w = run.w
    sc = ax.scatter(w.surface_points[:, 0], w.surface_points[:, 1], c=w.surface_score, cmap='viridis', s=12, vmin=0, vmax=1.2)
    draw_vesicles(ax, w.vesicles, colour='k', lw=0.6)
    frame(ax, w.field_px, 'surface points scored along each membrane')
    sc.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='score')

# plot_pickers: the two independent pickers' points above threshold
def plot_pickers(run, ax):
    w = run.w
    draw_vesicles(ax, w.vesicles, colour='0.7', lw=0.6)
    ax.scatter(w.picker_a[:, 0], w.picker_a[:, 1], s=55, facecolors='none', edgecolors=BLUE, linewidths=1.1, label=f'picker A (n={len(w.picker_a)})')
    ax.scatter(w.picker_b[:, 0], w.picker_b[:, 1], s=95, facecolors='none', edgecolors=ORANGE, linewidths=1.1, label=f'picker B (n={len(w.picker_b)})')
    frame(ax, w.field_px, 'two independent pickers above threshold')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.4)

# plot_consensus: points both pickers agree on, plus the one-picker-only rejects
def plot_consensus(run, ax):
    w = run.w
    draw_vesicles(ax, w.vesicles, colour='0.7', lw=0.6)
    if len(w.rejected_picks):
        ax.scatter(w.rejected_picks[:, 0], w.rejected_picks[:, 1], s=40, c=RED, marker='x', label=f'one picker only (n={len(w.rejected_picks)})')
    if len(w.consensus):
        ax.scatter(w.consensus[:, 0], w.consensus[:, 1], s=26, c=GREEN, label=f'consensus (n={len(w.consensus)})')
    frame(ax, w.field_px, 'reconcile: keep points both pickers agree on')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.4)

# plot_halfsets: half-set split (whole vesicle to one half) with rejected-surface decoys
def plot_halfsets(run, ax):
    w = run.w
    draw_vesicles(ax, w.vesicles, colour='0.7', lw=0.6)
    for half, colour in ((0, BLUE), (1, ORANGE)):
        sel = w.consensus_half == half
        if sel.any():
            ax.scatter(w.consensus[sel, 0], w.consensus[sel, 1], s=24, c=colour, label=f'half {half + 1} (n={int(sel.sum())})')
    if len(w.decoy_points):
        ax.scatter(w.decoy_points[:, 0], w.decoy_points[:, 1], s=30, c='0.5', marker='x', label=f'rejected-surface decoys (n={len(w.decoy_points)})')
    frame(ax, w.field_px, 'half-set split (whole vesicle -> one half) + decoys')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.4)

# plot_final: the final consensus particle set on the tomogram
def plot_final(run, ax):
    w = run.w
    ax.imshow(w.raw_tomogram, cmap=GREY)
    if len(w.consensus):
        ax.scatter(w.consensus[:, 0], w.consensus[:, 1], s=18, c=GREEN)
    frame(ax, w.field_px, f'final particle set on the tomogram (n={len(w.consensus)})')

# --- drivers ---
# _PANELS: ordered (filename stem, plot function) pairs for the pick walkthrough
_PANELS = [
    ('01_raw_tomogram', plot_raw),
    ('02_segmentation', plot_segmentation),
    ('03_surface_score', plot_surface_score),
    ('04_pickers', plot_pickers),
    ('05_consensus', plot_consensus),
    ('06_halfsets_decoys', plot_halfsets),
    ('07_final_particles', plot_final),
]

# save_panels: write each panel in _PANELS as its own PNG
def save_panels(run, outdir):
    for name, fn in _PANELS:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fn(run, ax)
        finish(fig, outdir / f'{name}.png')

# save_storyboard: lay every pick panel out on one 00_pick_storyboard.png figure
def save_storyboard(run, outdir):
    fig = plt.figure(figsize=(21, 12))
    grid = fig.add_gridspec(2, 4, hspace=0.2, wspace=0.2)
    slots = [grid[0, 0], grid[0, 1], grid[0, 2], grid[0, 3], grid[1, 0], grid[1, 1], grid[1, 2]]
    for slot, (_name, fn) in zip(slots, _PANELS):
        fn(run, fig.add_subplot(slot))
    fig.add_subplot(grid[1, 3]).set_axis_off()
    finish(fig, outdir / '00_pick_storyboard.png')

# render: run the mocked picking stage and write its figures, returning the output directory
def render(config: DiagramConfig) -> Path:
    return render_diagram(config, PickRun, save_panels, save_storyboard)
