'''
STAMP: detailed identify workflow diagram
'''

# Import external dependencies
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import rotate

# Import internal STAMP objects
from stamp.tools.diagram.classify import draw_null
from stamp.tools.diagram.utils.classify import class_label
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.render import DiagramRun, render_all_priors, render_diagram, render_mixed
from stamp.tools.diagram.utils.plotting import structure_projection_scatter
from stamp.utils.plotting.core import BLUE, GREEN, GREY, ORANGE, RED, bare, finish, style_axis, subgrid_axes
from stamp.utils.plotting.classify import class_average_grid
from stamp.identify.decoy_check import evaluate_decoy_control
from stamp.utils.plotting.identify import decoy_hist, score_heatmap

# ORIENTATIONS: in-plane angles (degrees) shown for each simulated density map
ORIENTATIONS = (0, 45, 90, 135)
# SIM_COLS: simulate-panel columns, one per orientation plus the actual-structure column
SIM_COLS = len(ORIENTATIONS) + 1

# IdentifyRun: run classify, register every class average against the candidate panel, and run the decoy control
class IdentifyRun(DiagramRun):
    def build(self):
        self.w.build_scene()
        self.w.pick()
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
    names = list(run.candidates)
    name_to_path = dict(zip(run.w.names, run.w.structure_paths))
    for row, name in enumerate(names):
        base = run.candidates[name]
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
        for col, angle in enumerate(ORIENTATIONS):
            ax = axes[row][col + 1]
            ax.imshow(base if angle == 0 else rotate(base, angle, reshape=False, order=1, mode='constant'), cmap=GREY)
            bare(ax)
            if row == 0:
                ax.set_title(f'sim {angle} deg', fontsize=8)

# plot_class_averages: class-average thumbnails titled with the best candidate per class
def plot_class_averages(run, axes):
    class_average_grid(axes, run.cluster_ids, run.averages, lambda c: f'{class_label(run.w.prior, c)}: {run.calls[c]["candidate"]}', title_size=7)

# plot_registration: correlation against rotation for the target class and each candidate, both contrasts' best curve, with the refined optimum marked
def plot_registration(run, ax):
    w = run.w
    for index, (name, result) in enumerate(w.registrations[w.target_cluster].items()):
        colour = (BLUE, ORANGE)[index % 2]
        ax.plot(result.coarse_angles, result.coarse_cc, '-o', ms=3, lw=1, color=colour, label=f'{name} (coarse scan)')
        ax.plot([a for a, _ in result.refined], [c for _, c in result.refined], 'D', ms=5, color=colour, mfc='none')
        ax.scatter([result.angle], [result.score], s=70, color=colour, zorder=3)
    style_axis(ax, f'registration of {class_label(w.prior, w.target_cluster)}: scan, then refine (diamonds), best pose (dot)', xlabel='rotation of the simulated map (deg)', ylabel='correlation', legend=True)

# plot_comparison_grid: every class average vs every candidate, best-fit contour overlaid
def plot_comparison_grid(run, fig, cell):
    names = list(run.candidates)
    inner = cell.subgridspec(len(run.cluster_ids), len(names), hspace=0.08, wspace=0.08)
    for r, cluster_id in enumerate(run.cluster_ids):
        average, call = run.averages[cluster_id], run.calls[cluster_id]
        for c, name in enumerate(names):
            ax = fig.add_subplot(inner[r, c])
            ax.imshow(average, cmap=GREY)
            ax.contour(run.fit_images[cluster_id][name], levels=3, colors=[GREEN], linewidths=0.7)
            won = name == call['candidate']
            ax.text(0.04, 0.96, f'CC {run.scores[cluster_id][name]:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=7, color='#1a7f37' if won else 'k', fontweight='bold' if won else 'normal', bbox=dict(boxstyle='round,pad=0.15', fc='w', ec='none', alpha=0.7))
            bare(ax)
            for spine in ax.spines.values():
                spine.set_edgecolor('#1a7f37' if won else '0.6')
                spine.set_linewidth(1.8 if won else 0.5)
            if r == 0:
                ax.set_title(name, fontsize=8)
            if c == 0:
                ax.set_ylabel(class_label(run.w.prior, cluster_id), fontsize=8)

# plot_score_matrix: class x candidate registration score heatmap
def plot_score_matrix(run, ax):
    score_heatmap(ax, run.cluster_ids, list(run.candidates), run.scores, title='registration score matrix')

# plot_decoy_null: decoy particles through the same polish, nearly all rejected against the null
def plot_decoy_null(run, ax):
    polish = run.w.decoy_polish[0] if run.w.decoy_polish else None
    if polish is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'decoys: no seed cluster reached the minimum size', ha='center', va='center', fontsize=8)
        return
    draw_null(ax, polish, np.zeros(len(polish.cc), dtype=bool), f'decoys through classify: {int((polish.labels == -1).sum())} of {len(polish.labels)} rejected (half 1)')

# plot_decoy_control: best registration score of real classes against decoy classes, with the verdict
def plot_decoy_control(run, ax):
    w = run.w
    if len(w.decoy_scores):
        decoy_hist(ax, np.array(w.real_scores), np.array(w.decoy_scores), w.decoy_passed)
    else:
        ax.hist(w.real_scores, bins=np.linspace(0, 1, 21), color=BLUE, alpha=0.8, label='real classes')
        style_axis(ax, 'decoy control: no decoy class survived', xlabel='best registration score', ylabel='classes', legend=True)
    ax.text(0.02, 0.97, f'{"PASS" if w.decoy_passed else "FAIL"}: {w.decoy_result.reason}', transform=ax.transAxes, va='top', ha='left', fontsize=6, wrap=True)

# plot_calls: text panel with the ranked call per class; identify reports the ranking, the decision is left to the reader
def plot_calls(run, ax):
    ax.set_axis_off()
    lines = ['ranked identification per class', '']
    for cluster_id in run.cluster_ids:
        call = run.calls[cluster_id]
        panel = '' if call['in_panel'] else '  (true species not in panel)'
        lines.append(f'{class_label(run.w.prior, cluster_id)}  ->  {call["candidate"]}   CC {call["cc"]:.2f}  gap {call["gap"]:.2f}   true: {call["true_species"]}{panel}')
    lines += ['', f'decoy control: {"PASS" if run.w.decoy_passed else "FAIL"}', f'best decoy class score: {max(run.w.decoy_scores):.2f}' if run.w.decoy_scores else 'no decoy class survived polish']
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left', fontsize=9, family='monospace')

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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_registration(run, ax)
    finish(fig, outdir / '03_registration.png')

    fig = plt.figure(figsize=(8, 2.4 * len(run.cluster_ids)))
    plot_comparison_grid(run, fig, fig.add_gridspec(1, 1)[0, 0])
    fig.suptitle('every class average vs every candidate (best-fit overlaid)', fontsize=10)
    finish(fig, outdir / '04_comparison_grid.png')

    fig, ax = plt.subplots(figsize=(5.5, 4))
    plot_score_matrix(run, ax)
    finish(fig, outdir / '05_score_matrix.png')

    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_decoy_null(run, ax)
    finish(fig, outdir / '06_decoy_null.png')

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_decoy_control(run, ax)
    finish(fig, outdir / '07_decoy_control.png')

    fig, ax = plt.subplots(figsize=(10, 3))
    plot_calls(run, ax)
    finish(fig, outdir / '08_calls.png')

# save_storyboard: lay every identify panel out on one 00_identify_storyboard.png figure
def save_storyboard(run, outdir):
    fig = plt.figure(figsize=(21, 16))
    grid = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.22)
    inner = grid[0, 0].subgridspec(2, SIM_COLS, hspace=0.1, wspace=0.1)
    plot_simulate(run, [[fig.add_subplot(inner[r, c]) for c in range(SIM_COLS)] for r in range(2)])
    plot_class_averages(run, subgrid_axes(fig, grid[0, 1], max(2, len(run.cluster_ids)), wspace=0.15))
    plot_registration(run, fig.add_subplot(grid[0, 2]))
    plot_comparison_grid(run, fig, grid[1, 0])
    plot_score_matrix(run, fig.add_subplot(grid[1, 1]))
    plot_decoy_null(run, fig.add_subplot(grid[1, 2]))
    plot_decoy_control(run, fig.add_subplot(grid[2, 0]))
    plot_calls(run, fig.add_subplot(grid[2, 1:3]))
    finish(fig, outdir / '00_identify_storyboard.png')

# render: run the mocked identify stage and write its figures, returning the output directory
def render(config: DiagramConfig) -> Path:
    return render_diagram(config, IdentifyRun, save_panels, save_storyboard)

# save_comparison: panels that differ by prior, one column per prior
def save_comparison(runs: dict, outdir: Path):
    rows = [plot_registration, plot_score_matrix, plot_decoy_control]
    fig = plt.figure(figsize=(7 * len(runs), 4.6 * len(rows)))
    grid = fig.add_gridspec(len(rows), len(runs), hspace=0.35, wspace=0.3)
    for col, run in enumerate(runs.values()):
        for row, fn in enumerate(rows):
            fn(run, fig.add_subplot(grid[row, col]))
    finish(fig, outdir / '00_identify_priors.png')

# render_all: identify diagram for every orientation prior, with a comparison figure
def render_all(config: DiagramConfig) -> Path:
    return render_all_priors(config, IdentifyRun, save_panels, save_storyboard, save_comparison)

# save_mixed: ranked calls of every prior group and one decoy control across all groups
def save_mixed(runs: dict, shared, outdir: Path):
    real = [score for run in runs.values() for score in run.w.real_scores]
    decoy = [score for run in runs.values() for score in run.w.decoy_scores]
    result = evaluate_decoy_control(real, decoy)
    fig, (ax_calls, ax_control) = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={'width_ratios': [1.6, 1]})
    ax_calls.set_axis_off()
    lines = ['ranked identification per prior group', '']
    for run in runs.values():
        for cluster_id in run.cluster_ids:
            call = run.calls[cluster_id]
            panel = '' if call['in_panel'] else '  (true species not in panel)'
            lines.append(f'{class_label(run.w.prior, cluster_id)}  ->  {call["candidate"]}   CC {call["cc"]:.2f}  gap {call["gap"]:.2f}   true: {call["true_species"]}{panel}')
    ax_calls.text(0.02, 0.98, '\n'.join(lines), transform=ax_calls.transAxes, va='top', ha='left', fontsize=9, family='monospace')
    if decoy:
        decoy_hist(ax_control, np.array(real), np.array(decoy), result.passed)
    else:
        ax_control.hist(real, bins=np.linspace(0, 1, 21), color=BLUE, alpha=0.8, label='real classes')
        style_axis(ax_control, 'decoy control: no decoy class survived', xlabel='best registration score', ylabel='classes', legend=True)
    ax_control.text(0.02, 0.97, f'{"PASS" if result.passed else "FAIL"} (one control across all priors): {result.reason}', transform=ax_control.transAxes, va='top', ha='left', fontsize=6, wrap=True)
    finish(fig, outdir / '00_identify_mixed.png')

# render_mixed_priors: identify diagram on one tomogram holding all three priors
def render_mixed_priors(config: DiagramConfig) -> Path:
    return render_mixed(config, IdentifyRun, save_panels, save_storyboard, save_mixed)
