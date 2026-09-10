'''
STAMP: overview of entire pipeline
'''

# Import external dependencies
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# Import internal STAMP objects
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.render import render_diagram
from stamp.tools.diagram.utils.plotting import BLUE, GREY, bare, class_average_grid, cluster_colour, decoy_hist, finish, scatter_groups, score_heatmap, show_segmentation, style_axis, subgrid_axes, tile_particles
from stamp.tools.diagram.utils.walkthrough import Walkthrough

# --- panels ---
# plot_ground_truth: ground-truth projection per species, tagged target / distractor / unknown
def plot_ground_truth(w, axes):
    for ax, (name, image) in zip(axes, w.structures):
        ax.imshow(image, cmap=GREY)
        tag = 'target' if name == w.target_name else 'distractor' if name == w.distractor_name else 'unknown'
        ax.set_title(f'{tag}: {name}', fontsize=8)
        bare(ax)
    for ax in axes[len(w.structures):]:
        ax.set_axis_off()

# plot_raw: raw tomogram slice
def plot_raw(w, ax):
    ax.imshow(w.raw_tomogram, cmap=GREY)
    ax.set_title('raw tomogram slice (vesicles + membrane proteins)', fontsize=9)
    bare(ax)

# plot_segmentation: membrane segmentation with consensus picks overlaid
def plot_segmentation(w, ax):
    show_segmentation(ax, w.segmentation, w.field_px, picks=w.consensus)
    ax.set_title('membrane segmentation + consensus picks', fontsize=9)
    ax.legend(fontsize=6, loc='upper right', framealpha=0.4)

# plot_extract: tiled grid of extracted particle subvolumes
def plot_extract(w, ax):
    n_show = min(25, len(w.particles))
    ax.imshow(tile_particles(w.particles[:n_show], w.box), cmap=GREY)
    ax.set_title(f'extracted particles (n={len(w.particles)}), classes unknown', fontsize=9)
    bare(ax)

# plot_classify: PCA embedding scatter coloured by KMeans cluster
def plot_classify(w, ax):
    groups = [(w.cluster_labels == cluster_id, cluster_colour(cluster_id), f'cluster {cluster_id}') for cluster_id in w.cluster_ids]
    scatter_groups(ax, w.embedding, groups, f'features -> PCA -> KMeans (k={len(w.cluster_ids)})')

# plot_class_averages: row of class-average thumbnails
def plot_class_averages(w, axes):
    class_average_grid(axes, w.cluster_ids, w.averages, lambda c: f'class {c} (n={w.class_counts[c]})')

# plot_identify: CC heatmap of each class against each candidate, with the call per class
def plot_identify(w, ax):
    labels = []
    for cluster_id in w.cluster_ids:
        call = w.calls[cluster_id]
        if call['identified']:
            labels.append(f'class {cluster_id} = {call["candidate"]}  ✓')
        else:
            labels.append(f'class {cluster_id} = no match  (true {call["true_species"]})')
    score_heatmap(ax, w.cluster_ids, list(w.candidates), w.score_matrix, row_labels=labels, title='identify: CC to each candidate; call per class')

# plot_decoy_control: real-vs-decoy fit-CC histogram
def plot_decoy_control(w, ax):
    decoy_hist(ax, w.real_scores, w.decoy_scores, w.decoy_passed)

# plot_refine: half-set FRC curve with the 0.143 threshold
def plot_refine(w, ax):
    ax.plot(w.frc_freq, w.frc_corr, color=BLUE, label='FRC (half A vs B)')
    ax.axhline(0.143, color='k', ls='--', lw=0.8, label='0.143')
    if np.isfinite(w.resolution):
        title = (f'refinement FRC: ~{w.resolution:.0f} A ({w.pixel_size:g} Å/px)')
    else:
        title = 'refinement FRC'
    ax.set_ylim(-0.1, 1.05)
    style_axis(ax, title, xlabel='spatial frequency (1/px)', ylabel='FRC', legend=True)


# plot_recovered: recovered map next to the target ground truth
def plot_recovered(w, axes):
    for ax, image, name in zip(axes, (w.recovered, w.target_truth), ('recovered from STAMP', f'ground truth: {w.target_name}')):
        ax.imshow(image, cmap=GREY)
        ax.set_title(name, fontsize=9)
        bare(ax)


# --- drivers ---
# save_individual_panels: write each stage panel as a numbered PNG
def save_individual_panels(w, outdir: Path):
    n = w.n_classes
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
    plot_ground_truth(w, np.atleast_1d(axes))
    finish(fig, outdir / '01_ground_truth.png')

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_raw(w, ax)
    finish(fig, outdir / '02_raw_tomogram.png')

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    plot_segmentation(w, ax)
    finish(fig, outdir / '03_segmentation_picks.png')

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_extract(w, ax)
    finish(fig, outdir / '04_extract.png')

    fig, ax = plt.subplots(figsize=(5.5, 4))
    plot_classify(w, ax)
    finish(fig, outdir / '05_classify.png')

    fig, axes = plt.subplots(1, max(2, n), figsize=(2.6 * max(2, n), 3))
    plot_class_averages(w, np.atleast_1d(axes))
    finish(fig, outdir / '06_class_averages.png')

    fig, ax = plt.subplots(figsize=(6.5, 4))
    plot_identify(w, ax)
    finish(fig, outdir / '07_identify.png')

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_decoy_control(w, ax)
    finish(fig, outdir / '08_decoy_control.png')

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_refine(w, ax)
    finish(fig, outdir / '09_refine_frc.png')

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    plot_recovered(w, axes)
    finish(fig, outdir / '10_recovered.png')

# save_storyboard: lay every stage panel out on one 00_storyboard.png figure
def save_storyboard(w, outdir: Path):
    n = max(2, w.n_classes)
    fig = plt.figure(figsize=(21, 13))
    grid = fig.add_gridspec(3, 4, hspace=0.32, wspace=0.28, height_ratios=[1.0, 0.72, 1.0])

    plot_raw(w, fig.add_subplot(grid[0, 0]))
    plot_segmentation(w, fig.add_subplot(grid[0, 1]))
    plot_extract(w, fig.add_subplot(grid[0, 2]))
    plot_classify(w, fig.add_subplot(grid[0, 3]))

    plot_class_averages(w, subgrid_axes(fig, grid[1, :], n))

    plot_identify(w, fig.add_subplot(grid[2, 0]))
    plot_decoy_control(w, fig.add_subplot(grid[2, 1]))
    plot_refine(w, fig.add_subplot(grid[2, 2]))
    inner = grid[2, 3].subgridspec(2, 1, hspace=0.4)
    plot_recovered(w, [fig.add_subplot(inner[0]), fig.add_subplot(inner[1])])

    extras = [name for name in w.names if name not in (w.target_name, w.distractor_name)]
    finish(fig, outdir / '00_storyboard.png')

# _build: run the mocked pipeline and return the finished Walkthrough
def _build(config: DiagramConfig) -> Walkthrough:
    w = Walkthrough(config)
    w.run()
    return w

# render: run the mocked pipeline and write the overview figures, returning the output directory
def render(config: DiagramConfig) -> Path:
    return render_diagram(config, _build, save_individual_panels, save_storyboard)
