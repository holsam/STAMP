'''
STAMP: detailed native picker workflow diagram
'''

# Import external dependencies
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# Import internal STAMP objects
from stamp.tools.diagram.utils.config import DiagramConfig, NativePickerConfig
from stamp.tools.diagram.utils.render import DiagramRun, render_diagram
from stamp.tools.diagram.utils.native import run_native_picker
from stamp.tools.diagram.utils.plotting import BLUE, GREEN, GREY, ORANGE, RED, bare, draw_vesicles, finish, frame, show_segmentation

# NativeRun: run the native picker on the mock scene and score its picks against the ground truth
class NativeRun(DiagramRun):
    def build(self):
        self.w.build_scene()
        self.diagram_config = self.config
        self.config = NativePickerConfig.from_diagram_config(self.diagram_config)
        self.r = run_native_picker(self.w.raw_tomogram, self.w.segmentation, self.w.vesicles, self.config)
        self.field_px = self.w.field_px
        self.true_xy = self.w.true_xy
        # precision / recall of the native picks against the ground truth
        capture = 0.5 * self.w.box
        picks = self.r['picks']
        if len(picks) and len(self.true_xy):
            dist = np.linalg.norm(picks[:, None, :] - self.true_xy[None, :, :], axis=2)
            self.tp = int((dist.min(axis=1) < capture).sum())
            self.recall = (dist.min(axis=0) < capture).sum() / len(self.true_xy)
        else:
            self.tp, self.recall = 0, 0.0
        self.precision = self.tp / max(len(picks), 1)


# --- panels ---
# plot_inputs: raw tomogram next to the membrane segmentation
def plot_inputs(run, axes):
    ax_raw, ax_seg = axes
    ax_raw.imshow(run.w.raw_tomogram, cmap=GREY)
    frame(ax_raw, run.field_px, 'input: raw tomogram')
    show_segmentation(ax_seg, run.w.segmentation, run.field_px)
    frame(ax_seg, run.field_px, 'input: membrane segmentation')

# plot_blank: tomogram with the membrane blanked to the background median
def plot_blank(run, ax):
    ax.imshow(run.r['blanked'], cmap=GREY)
    frame(ax, run.field_px, 'membrane blanked to background median\n(shell then scores only off-membrane density)')

# plot_surface: the extracted + downsampled surface points with outward normals
def plot_surface(run, ax):
    ax.imshow(run.r['blanked'], cmap=GREY, alpha=0.5)
    v = run.r['vertices']
    n = run.r['normals']
    ax.scatter(v[:, 0], v[:, 1], s=6, c=BLUE)
    step = max(1, len(v) // 220)
    ax.quiver(v[::step, 0], v[::step, 1], n[::step, 0], -n[::step, 1], color=ORANGE, scale=30, width=0.003, headwidth=4)
    frame(ax, run.field_px, f'surface extracted + downsampled ({len(v)} points, {run.config.surface_spacing_angstrom:g} Å spacing)')

# plot_sampling: zoom on one vesicle, showing the density shell sampled along each normal
def plot_sampling(run, ax):
    v = run.w.vesicles[0]
    pad = v['r'] + run.r['off_max_px'] + 12
    ax.imshow(run.r['blanked'], cmap=GREY)
    ax.add_patch(plt.Circle((v['cx'], v['cy']), v['r'], fill=False, edgecolor=GREEN, lw=1.2))
    for scale, style in ((run.r['off_min_px'], ':'), (run.r['off_max_px'], '--')):
        for sign in (1, -1):
            ax.add_patch(plt.Circle((v['cx'], v['cy']), v['r'] + sign * scale, fill=False, edgecolor=ORANGE, lw=0.8, ls=style))
    offsets = np.linspace(run.r['off_min_px'], run.r['off_max_px'], run.config.n_samples)
    for theta in np.linspace(0, 2 * np.pi, 14, endpoint=False):
        d = np.array([np.cos(theta), np.sin(theta)])
        base = np.array([v['cx'], v['cy']]) + v['r'] * d
        for sign in (1, -1):
            pts = base[None, :] + sign * offsets[:, None] * d[None, :]
            ax.scatter(pts[:, 0], pts[:, 1], s=10, c=BLUE if sign == 1 else RED)
    ax.set_xlim(v['cx'] - pad, v['cx'] + pad)
    ax.set_ylim(v['cy'] + pad, v['cy'] - pad)
    ax.set_aspect('equal')
    ax.set_title(f'sample density shell along the normal, both faces\n'
                 f'offsets {run.config.offset_min_angstrom:g}-'
                 f'{run.config.offset_max_angstrom:g} A, '
                 f'{run.config.n_samples} samples', fontsize=9)
    bare(ax)


# plot_score_faces: robust score per surface point for the inner and outer faces
def plot_score_faces(run, fig, cell):
    inner = cell.subgridspec(1, 2, wspace=0.15)
    limit = max(3.0, np.abs(run.r['all_scores']).max())
    for i, face in enumerate(run.r['faces']):
        ax = fig.add_subplot(inner[i])
        sc = ax.scatter(face['points'][:, 0], face['points'][:, 1], c=face['scores'], cmap='coolwarm', s=10, vmin=-limit, vmax=limit)
        draw_vesicles(ax, run.w.vesicles, colour='k', lw=0.5)
        frame(ax, run.field_px, f'{"outer" if face["direction"] == 1 else "inner"} face score')
        if i == 1:
            cbar = sc.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.06)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label('robust score (MAD)', fontsize=6)

# plot_histogram: robust score distribution over both faces with the n_mad threshold
def plot_histogram(run, ax):
    ax.hist(run.r['all_scores'], bins=40, color='#b0b0b0')
    ax.axvline(run.config.n_mad, color=RED, ls='--', lw=1.0, label=f'threshold n_mad = {run.config.n_mad:g}')
    ax.set_title('robust score distribution (both faces)', fontsize=9)
    ax.set_xlabel('score (MAD units)', fontsize=8)
    ax.set_ylabel('surface points', fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

# plot_threshold_nms: points above threshold and the subset kept after NMS
def plot_threshold_nms(run, ax):
    above = run.r['all_points'][run.r['above_mask']]
    ax.imshow(run.w.raw_tomogram, cmap=GREY, alpha=0.55)
    if len(above):
        ax.scatter(above[:, 0], above[:, 1], s=22, facecolors='none', edgecolors=ORANGE, linewidths=0.9, label=f'above threshold (n={len(above)})')
    picks = run.r['picks']
    if len(picks):
        ax.scatter(picks[:, 0], picks[:, 1], s=26, c=GREEN, label=f'kept after NMS (n={len(picks)})')
    frame(ax, run.field_px, f'threshold then NMS (min separation {run.config.min_particle_distance_angstrom:g} Å)')
    if len(above) or len(picks):
        ax.legend(fontsize=6, loc='upper right', framealpha=0.4)

# plot_final: final picks with orientation from the surface normal, plus precision / recall
def plot_final(run, ax):
    ax.imshow(run.w.raw_tomogram, cmap=GREY)
    picks = run.r['picks']
    n = run.r['pick_normals']
    if len(picks):
        ax.quiver(picks[:, 0], picks[:, 1], n[:, 0], -n[:, 1], color=ORANGE, scale=22, width=0.004, headwidth=4)
        ax.scatter(picks[:, 0], picks[:, 1], s=18, c=GREEN)
    frame(ax, run.field_px, f'picks + orientation from normal  (prec {run.precision:.2f}, recall {run.recall:.2f})')

# --- drivers ---
# save_panels: write each stage panel as a numbered PNG
def save_panels(run, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    plot_inputs(run, axes)
    finish(fig, outdir / '01_inputs.png')

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_blank(run, ax)
    finish(fig, outdir / '02_blank_membrane.png')

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_surface(run, ax)
    finish(fig, outdir / '03_surface_normals.png')

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_sampling(run, ax)
    finish(fig, outdir / '04_normal_sampling.png')

    fig = plt.figure(figsize=(11, 5))
    plot_score_faces(run, fig, fig.add_gridspec(1, 1)[0, 0])
    finish(fig, outdir / '05_score_faces.png')

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_histogram(run, ax)
    finish(fig, outdir / '06_score_histogram.png')

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_threshold_nms(run, ax)
    finish(fig, outdir / '07_threshold_nms.png')

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_final(run, ax)
    finish(fig, outdir / '08_picks_orientation.png')

# save_storyboard: lay every native-picker panel out on one storyboard figure
def save_storyboard(run, outdir):
    fig = plt.figure(figsize=(23, 11))
    grid = fig.add_gridspec(2, 4, hspace=0.25, wspace=0.3)

    inner = grid[0, 0].subgridspec(2, 1, hspace=0.3)
    plot_inputs(run, [fig.add_subplot(inner[0]), fig.add_subplot(inner[1])])
    plot_blank(run, fig.add_subplot(grid[0, 1]))
    plot_surface(run, fig.add_subplot(grid[0, 2]))
    plot_sampling(run, fig.add_subplot(grid[0, 3]))

    plot_score_faces(run, fig, grid[1, 0])
    plot_histogram(run, fig.add_subplot(grid[1, 1]))
    plot_threshold_nms(run, fig.add_subplot(grid[1, 2]))
    plot_final(run, fig.add_subplot(grid[1, 3]))

    finish(fig, outdir / '00_native_picker_storyboard.png')

# render: run the mocked native picker and write its figures, returning the output directory
def render(config: DiagramConfig) -> Path:
    return render_diagram(config, NativeRun, save_panels, save_storyboard)
