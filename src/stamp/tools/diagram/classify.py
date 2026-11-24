'''
STAMP: detailed classify workflow diagram
'''

# Import external dependencies
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.tools.diagram.utils.classify import class_label
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.render import DiagramRun, render_diagram
from stamp.utils.plotting.core import BLUE, CLUSTER_COLOURS, GREY, ORANGE, RED, bare, cluster_colour, finish, scatter_groups, style_axis, subgrid_axes, tile_particles
from stamp.utils.plotting.classify import class_average_grid, scatter_labels

# MAX_REFS: references drawn per row in polish panels
MAX_REFS = 6

# ClassifyRun: run classify through features, seed clustering and polish, each half on its own
class ClassifyRun(DiagramRun):
    def build(self):
        self.w.build_scene()
        self.w.pick()
        self.w.extract_particles()
        self.w.classify()
        self.w.class_averages()
        self.half_a, self.half_b = self.w.halves

# --- panels ---
# plot_subvolumes: tiled grid of extracted subvolumes, noisy and missing-wedge filtered
def plot_subvolumes(run, ax):
    w = run.w
    n_show = min(30, len(w.particles))
    ax.imshow(tile_particles(w.particles[:n_show], w.box, columns=6), cmap=GREY)
    ax.set_title(f'extracted subvolumes ({n_show} of {len(w.particles)}), pose unknown beyond the prior', fontsize=8)
    bare(ax)

# plot_features: feature matrix heatmap sorted by true species
def plot_features(run, ax):
    w = run.w
    order = np.argsort(w.particle_class, kind='stable')
    ax.imshow(w.standardised[order], aspect='auto', cmap='coolwarm', vmin=-3, vmax=3)
    ax.set_title(f'features invariant to what the {w.prior.value} prior leaves open', fontsize=8)
    ax.set_xlabel('feature', fontsize=8)
    ax.set_ylabel('particle (sorted by true species)', fontsize=8)
    ax.tick_params(labelsize=6)

# plot_feature_means: class-mean standardised feature vector per true species
def plot_feature_means(run, ax):
    w = run.w
    for class_id, name in enumerate(w.names):
        members = w.particle_class == class_id
        if members.any():
            ax.plot(w.standardised[members].mean(axis=0), color=cluster_colour(class_id), alpha=0.8, label=name)
    style_axis(ax, 'class-mean standardised feature', xlabel='feature', ylabel='z-score', legend=True)

# plot_pca: whitened PCA embedding of half A coloured by true species, with variance explained in title
def plot_pca(run, ax):
    w, half = run.w, run.half_a
    groups = [(w.particle_class[half.index] == class_id, cluster_colour(class_id), name) for class_id, name in enumerate(w.names)]
    groups.append((w.particle_class[half.index] < 0, '#b0b0b0', 'junk pick'))
    scatter_groups(ax, half.embedding[:, :2], groups, f'half A: PCA ({half.explained.sum():.0%} variance in {len(half.explained)} PCs), true species', size=18)

# plot_seeds: half embedding coloured by seed cluster
def plot_seeds(run, ax, half, name):
    n_noise = int((half.seed_labels == -1).sum())
    scatter_labels(ax, half.embedding[:, :2], half.seed_labels, f'half {name}: {run.config.cluster_method} seeds ({len(set(half.seed_labels) - {-1})} clusters, {n_noise} noise)')

# plot_search_space: rotations alignment tries for prior, with shift range
def plot_search_space(run, ax):
    space = run.w.space
    theta = np.deg2rad(space.angles)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ec='0.7', lw=0.8))
    ax.scatter(np.cos(theta), np.sin(theta), s=18, c=BLUE)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    shifts = f'axial shift only, +-{space.max_shift} px' if space.axial_only else f'shift +-{space.max_shift} px in x and y'
    what = {OrientationPrior.NORMAL: 'azimuth about the normal', OrientationPrior.AXIS: 'polarity flip (0 or 180 deg)', OrientationPrior.NONE: 'full rotation, finer grid'}[space.prior]
    ax.set_title(f'search space: {what}, {len(space.angles)} rotations, {shifts}', fontsize=7)
    bare(ax)

# plot_wedges: observed Fourier directions of three particles, which follow each particle's orientation
def plot_wedges(run, axes):
    w = run.w
    for ax, index in zip(axes, np.linspace(0, len(w.particles) - 1, len(axes)).astype(int)):
        ax.imshow(np.fft.fftshift(w.particle_masks[index]), cmap=GREY, vmin=0, vmax=1)
        ax.set_title(f'wedge, particle {index}' + (f'\n(frame {w.particle_phi[index]:.0f} deg)' if w.prior is not OrientationPrior.NONE else '\n(fixed)'), fontsize=6)
        bare(ax)

# plot_references: seed references (top) and polished references (bottom) of half A
def plot_references(run, fig, cell):
    polish = run.half_a.polish
    inner = cell.subgridspec(2, MAX_REFS, hspace=0.25, wspace=0.08)
    for row, (name, refs) in enumerate((('seed', polish.seed_references), ('polished', polish.references))):
        for col in range(MAX_REFS):
            ax = fig.add_subplot(inner[row, col])
            if col < len(refs):
                ax.imshow(refs[col], cmap=GREY)
                if col == 0:
                    ax.set_ylabel(name, fontsize=7)
            else:
                ax.set_axis_off()
            if col < len(refs):
                bare(ax)
            if row == 0 and col == MAX_REFS // 2 - 1:
                ax.set_title('half A references: seeds, then after polish rounds', fontsize=7, loc='left')

# plot_merge: agreement between references against their half-split ceilings, merged pairs outlined
def plot_merge(run, ax):
    polish = run.half_a.polish
    similarity = polish.similarity.copy()
    ax.imshow(similarity, cmap='viridis', vmin=0, vmax=1)
    for i in range(len(similarity)):
        for j in range(len(similarity)):
            ax.text(j, i, f'{similarity[i, j]:.2f}' if i != j else f'ceil\n{polish.ceilings[i]:.2f}', ha='center', va='center', fontsize=6, color='w')
    for i, j in polish.merged:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=RED, lw=2))
    ax.set_xticks(range(len(similarity)))
    ax.set_yticks(range(len(similarity)))
    ax.tick_params(labelsize=6)
    merges = f'{len(polish.merged)} merged' if polish.merged else 'none merged'
    ax.set_title(f'merge: reference agreement vs half-split ceiling ({merges})', fontsize=7)

# draw_null: correlation of particles with their reference against the phase-randomised null, with the two rejection bars
def draw_null(ax, polish, junk, title):
    bins = np.linspace(0, 1, 41)
    ax.hist(polish.null_best, bins=bins, color='0.6', alpha=0.8, label='phase-randomised null')
    ax.hist(polish.cc[~junk], bins=bins, color=BLUE, alpha=0.7, label='particles')
    if junk.any():
        ax.hist(polish.cc[junk], bins=bins, color=RED, alpha=0.9, label='junk picks')
    ax.axvline(polish.particle_bar, color='k', ls='--', lw=0.9, label=f'reject below {polish.particle_bar:.2f}')
    ax.axvline(polish.reference_bar, color=ORANGE, ls=':', lw=1.2, label=f'reference bar {polish.reference_bar:.2f}')
    style_axis(ax, title, xlabel='correlation with best reference', ylabel='particles', legend=True)

# plot_null: half A particles against the null
def plot_null(run, ax):
    polish = run.half_a.polish
    draw_null(ax, polish, run.w.particle_class[run.half_a.index] < 0, f'null rejection: {int((polish.labels == -1).sum())} of {len(polish.labels)} half A particles rejected')

# plot_halves: half A references (top) and matched half B references moved into frame of A (bottom)
def plot_halves(run, fig, cell):
    w = run.w
    inner = cell.subgridspec(2, MAX_REFS, hspace=0.25, wspace=0.08)
    for col, cluster_id in enumerate(w.cluster_ids[:MAX_REFS]):
        for row, image in enumerate(w.half_averages[cluster_id]):
            ax = fig.add_subplot(inner[row, col])
            ax.imshow(image, cmap=GREY)
            bare(ax)
            if col == 0:
                ax.set_ylabel(f'half {"AB"[row]}', fontsize=7)
            if row == 0 and col == MAX_REFS // 2 - 1:
                ax.set_title('class averages per half, B matched to A by reference', fontsize=7, loc='left')

# plot_class_averages: final class averages, wedge-weighted over both halves, with member counts
def plot_class_averages(run, axes):
    w = run.w
    class_average_grid(axes, w.cluster_ids, w.averages, lambda c: f'{class_label(w.prior, c)} (n={w.class_counts[c]})')

# --- drivers ---
# _n_classes: columns needed for a row of class averages
def _n_classes(run) -> int:
    return max(2, len(run.w.cluster_ids))

# save_panels: write each stage panel as a numbered PNG
def save_panels(run, outdir):
    single = [('subvolumes', plot_subvolumes, (6, 5)), ('features', plot_features, (6, 4.5)), ('feature_means', plot_feature_means, (6, 4)), ('pca', plot_pca, (5.5, 4.5)),
              ('seeds_half_a', lambda r, ax: plot_seeds(r, ax, r.half_a, 'A'), (5.5, 4.5)), ('seeds_half_b', lambda r, ax: plot_seeds(r, ax, r.half_b, 'B'), (5.5, 4.5)),
              ('search_space', plot_search_space, (4.5, 4.5)), ('merge', plot_merge, (5, 4.5)), ('null', plot_null, (6, 4.5))]
    for number, (name, fn, size) in enumerate(single, start=1):
        fig, ax = plt.subplots(figsize=size)
        fn(run, ax)
        finish(fig, outdir / f'{number:02d}_{name}.png')
    number = len(single) + 1
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))
    plot_wedges(run, axes)
    finish(fig, outdir / f'{number:02d}_wedges.png')
    fig = plt.figure(figsize=(9, 3.4))
    plot_references(run, fig, fig.add_gridspec(1, 1)[0, 0])
    finish(fig, outdir / f'{number + 1:02d}_references.png')
    fig = plt.figure(figsize=(9, 3.4))
    plot_halves(run, fig, fig.add_gridspec(1, 1)[0, 0])
    finish(fig, outdir / f'{number + 2:02d}_halves.png')
    n = _n_classes(run)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
    plot_class_averages(run, np.atleast_1d(axes))
    finish(fig, outdir / f'{number + 3:02d}_class_averages.png')

# save_storyboard: lay every classify panel out on one 00_classify_storyboard.png figure
def save_storyboard(run, outdir):
    fig = plt.figure(figsize=(22, 17))
    grid = fig.add_gridspec(4, 4, hspace=0.38, wspace=0.28, height_ratios=[1, 1, 1, 0.55])
    plot_subvolumes(run, fig.add_subplot(grid[0, 0]))
    plot_features(run, fig.add_subplot(grid[0, 1]))
    plot_pca(run, fig.add_subplot(grid[0, 2]))
    plot_seeds(run, fig.add_subplot(grid[0, 3]), run.half_a, 'A')
    plot_search_space(run, fig.add_subplot(grid[1, 0]))
    plot_wedges(run, subgrid_axes(fig, grid[1, 1], 3, wspace=0.1))
    plot_references(run, fig, grid[1, 2:4])
    plot_merge(run, fig.add_subplot(grid[2, 0]))
    plot_null(run, fig.add_subplot(grid[2, 1]))
    plot_halves(run, fig, grid[2, 2:4])
    plot_class_averages(run, subgrid_axes(fig, grid[3, :], _n_classes(run)))
    finish(fig, outdir / '00_classify_storyboard.png')

# render: run mocked classify stage and write its figures, returning output directory
def render(config: DiagramConfig) -> Path:
    return render_diagram(config, ClassifyRun, save_panels, save_storyboard)
