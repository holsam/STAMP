'''
STAMP: detailed identify workflow diagram
'''

# Import external dependencies
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import rotate

# Import internal STAMP objects
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.render import DiagramRun, render_diagram
from stamp.tools.diagram.utils.plotting import GREEN, GREY, bare, class_average_grid, decoy_hist, finish, score_heatmap, structure_projection_scatter, subgrid_axes

# ORIENTATIONS: in-plane angles (degrees) shown for each simulated density map
ORIENTATIONS = (0, 45, 90, 135)
# SIM_COLS: simulate-panel columns, one per orientation plus the actual-structure column
SIM_COLS = len(ORIENTATIONS) + 1

# IdentifyRun: run classify + identify + the decoy control on the mock mixture
class IdentifyRun(DiagramRun):
    def build(self):
        self.w.build_scene()
        self.w.extract_particles()
        self.w.classify()
        self.w.class_averages()
        self.w.identify()
        self.w.decoy_control()

        self.candidates = self.w.candidates
        self.cluster_ids = self.w.cluster_ids
        self.averages = self.w.averages
        self.scores = self.w.score_matrix
        self.fit_images = self.w.fit_images
        self.calls = self.w.calls

# --- panels ---
# plot_simulate: actual atoms in column 0, then the simulated density map at each orientation
def plot_simulate(run, axes):
    names = [n for n in run.candidates if n != 'decoy_model']
    name_to_path = dict(zip(run.w.names, run.w.structure_paths))
    for row, name in enumerate(names):
        base = run.candidates[name]
        # column 0: the actual atomic structure
        ax0 = axes[row][0]
        path = name_to_path.get(name)
        if path is not None:
            structure_projection_scatter(ax0, path, run.w.box, run.w.pixel_size)
        else:
            ax0.imshow(base, cmap=GREY)
            bare(ax0)
        ax0.set_ylabel(name, fontsize=9)
        if row == 0:
            ax0.set_title('structure', fontsize=8)
        # columns 1..: simulated density map, rotated
        for col, angle in enumerate(ORIENTATIONS):
            ax = axes[row][col + 1]
            image = base if angle == 0 else rotate(base, angle, reshape=False, order=1, mode='constant')
            ax.imshow(image, cmap=GREY)
            bare(ax)
            if row == 0:
                ax.set_title(f'sim {angle} deg', fontsize=8)

# plot_class_averages: class-average thumbnails titled with the call per class
def plot_class_averages(run, axes):
    def title(cluster_id):
        call = run.calls[cluster_id]
        tag = call['candidate'] if call['identified'] else 'no match'
        return f'cl{cluster_id}: {tag}'
    class_average_grid(axes, run.cluster_ids, run.averages, title, title_size=7)

# plot_comparison_grid: every class average vs every candidate, best-fit contour overlaid
def plot_comparison_grid(run, fig, cell):
    names = list(run.candidates)
    inner = cell.subgridspec(len(run.cluster_ids), len(names), hspace=0.08, wspace=0.08)
    for r, cluster_id in enumerate(run.cluster_ids):
        average = run.averages[cluster_id]
        call = run.calls[cluster_id]
        for c, name in enumerate(names):
            ax = fig.add_subplot(inner[r, c])
            ax.imshow(average, cmap=GREY)
            ax.contour(run.fit_images[cluster_id][name], levels=3, colors=[GREEN], linewidths=0.7)
            cc = run.scores[cluster_id][name]
            won = call['identified'] and name == call['candidate']
            ax.text(0.04, 0.96, f'CC {cc:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=7, color='#1a7f37' if won else 'k', fontweight='bold' if won else 'normal', bbox=dict(boxstyle='round,pad=0.15', fc='w', ec='none', alpha=0.7))
            bare(ax)
            for spine in ax.spines.values():
                spine.set_edgecolor('#1a7f37' if won else '0.6')
                spine.set_linewidth(1.8 if won else 0.5)
            if r == 0:
                ax.set_title(name, fontsize=8)
            if c == 0:
                ax.set_ylabel(f'class {cluster_id}', fontsize=8)

# plot_score_matrix: class x candidate cross-correlation heatmap
def plot_score_matrix(run, ax):
    score_heatmap(ax, run.cluster_ids, list(run.candidates), run.scores, title='cross-correlation matrix')

# plot_decoy_control: real-vs-decoy fit-CC histogram
def plot_decoy_control(run, ax):
    w = run.w
    decoy_hist(ax, w.real_scores, w.decoy_scores, w.decoy_passed)

# plot_calls: text panel listing the identification call and reasoning per class
def plot_calls(run, ax):
    ax.set_axis_off()
    lines = ['identification calls', '']
    for cluster_id in run.cluster_ids:
        call = run.calls[cluster_id]
        if call['identified']:
            lines.append(
                f'class {cluster_id}  ->  {call["candidate"]}'
                f'   CC {call["cc"]:.2f}  gap {call["gap"]:.2f}   [MATCH]'
            )
        else:
            why = ('best is the decoy' if call['best_name'] == 'decoy_model'
                   else f'CC {call["cc"]:.2f} / gap {call["gap"]:.2f} below threshold')
            panel = '' if call['in_panel'] else '  (true species not in panel)'
            lines.append(
                f'class {cluster_id}  ->  no confident match   {why}'
                f'   true: {call["true_species"]}{panel}'
            )
    lines += ['',
              f'thresholds: CC >= {run.config.identify_min_cc}, '
              f'gap >= {run.config.identify_min_gap}',
              f'decoy control: {"PASS" if run.w.decoy_passed else "FAIL"}']
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left', fontsize=10, family='monospace')

# --- drivers ---
# save_panels: write each stage panel as a numbered PNG
def save_panels(run, outdir):
    fig, axes = plt.subplots(2, SIM_COLS, figsize=(2.1 * SIM_COLS, 4.5))
    plot_simulate(run, axes)
    fig.suptitle('actual structure, then simulated density map over in-plane angle', fontsize=10)
    finish(fig, outdir / '01_simulate.png')

    n = max(2, len(run.cluster_ids))
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
    plot_class_averages(run, np.atleast_1d(axes))
    finish(fig, outdir / '02_class_averages.png')

    fig = plt.figure(figsize=(8, 2.4 * len(run.cluster_ids)))
    plot_comparison_grid(run, fig, fig.add_gridspec(1, 1)[0, 0])
    fig.suptitle('every class average vs every candidate (best-fit overlaid)', fontsize=10)
    finish(fig, outdir / '03_comparison_grid.png')

    fig, ax = plt.subplots(figsize=(5.5, 4))
    plot_score_matrix(run, ax)
    finish(fig, outdir / '04_score_matrix.png')

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_decoy_control(run, ax)
    finish(fig, outdir / '05_decoy_control.png')

    fig, ax = plt.subplots(figsize=(9, 3))
    plot_calls(run, ax)
    finish(fig, outdir / '06_calls.png')

# save_storyboard: lay every identify panel out on one 00_identify_storyboard.png figure
def save_storyboard(run, outdir):
    fig = plt.figure(figsize=(21, 11))
    grid = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

    inner = grid[0, 0].subgridspec(2, SIM_COLS, hspace=0.1, wspace=0.1)
    axes = [[fig.add_subplot(inner[r, c]) for c in range(SIM_COLS)] for r in range(2)]
    plot_simulate(run, axes)

    n = max(2, len(run.cluster_ids))
    plot_class_averages(run, subgrid_axes(fig, grid[0, 1], n, wspace=0.15))

    plot_comparison_grid(run, fig, grid[0, 2])

    plot_score_matrix(run, fig.add_subplot(grid[1, 0]))
    plot_decoy_control(run, fig.add_subplot(grid[1, 1]))
    plot_calls(run, fig.add_subplot(grid[1, 2]))

    finish(fig, outdir / '00_identify_storyboard.png')

# render: run the mocked identify stage and write its figures, returning the output directory
def render(config: DiagramConfig) -> Path:
    return render_diagram(config, IdentifyRun, save_panels, save_storyboard)
