'''
STAMP: detailed classify workflow diagram
'''

# Import external dependencies
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import HDBSCAN

# Import internal STAMP objects
from stamp.tools.diagram.utils.config import DiagramConfig
from stamp.tools.diagram.utils.render import DiagramRun, render_diagram
from stamp.tools.diagram.utils.classify import radial_profile
from stamp.tools.diagram.utils.plotting import CLUSTER_COLOURS, GREY, bare, class_average_grid, cluster_colour, finish, scatter_groups, scatter_labels, style_axis, subgrid_axes, tile_particles

# ClassifyRun: run classify through KMeans, then repeat the clustering with HDBSCAN
class ClassifyRun(DiagramRun):
    def build(self):
        config = self.config
        self.w.build_scene()
        self.w.extract_particles()
        self.w.classify(n_clusters=config.kmeans_k)
        self.w.class_averages()

        self.particles = self.w.particles
        self.true_class = self.w.particle_class
        self.box = self.w.box
        self.names = self.w.names
        self.n_radial = len(radial_profile(self.particles[0]))
        self.features = self.w.features
        self.standardised = self.w.standardised
        self.pca = self.w.pca
        self.embedding = self.w.embedding
        self.kmeans_labels = self.w.cluster_labels

        mcs = max(3, min(config.hdbscan_min_cluster_size, len(self.particles) // 4))
        raw_hdb = HDBSCAN(min_cluster_size=mcs, min_samples=mcs, copy=True).fit_predict(self.w.pca_embedding)
        self.hdbscan_labels = self._order(raw_hdb)

    # _order: relabel real clusters by dominant true class, keeping noise (-1)
    def _order(self, labels):
        labels = np.asarray(labels).copy()
        real = [l for l in sorted(set(labels)) if l != -1]
        if not real:
            return labels
        key = {l: int(np.bincount(self.true_class[labels == l]).argmax()) for l in real}
        order = sorted(real, key=lambda l: (key[l], -int(np.sum(labels == l))))
        remap = {old: new for new, old in enumerate(order)}
        return np.array([remap.get(l, -1) for l in labels])

# --- panels ---
# plot_subvolumes: tiled grid of extracted subvolumes
def plot_subvolumes(run, ax):
    n_show = min(30, len(run.particles))
    ax.imshow(tile_particles(run.particles[:n_show], run.box, columns=6), cmap=GREY)
    ax.set_title(f'extracted subvolumes ({n_show} of {len(run.particles)})', fontsize=9)
    bare(ax)


# plot_features: feature matrix heatmap plus radial-profile curves per class
def plot_features(run, axes):
    ax_matrix, ax_curves = axes
    order = np.argsort(run.true_class, kind='stable')
    ax_matrix.imshow(run.standardised[order], aspect='auto', cmap='coolwarm', vmin=-3, vmax=3)
    ax_matrix.axvline(run.n_radial - 0.5, color='k', lw=0.8)
    ax_matrix.set_title('feature matrix (rows sorted by true class)', fontsize=9)
    ax_matrix.set_xlabel('radial profile | annular Fourier power', fontsize=8)
    ax_matrix.set_ylabel('particle', fontsize=8)
    ax_matrix.tick_params(labelsize=6)

    for class_id, name in enumerate(run.names):
        members = np.where(run.true_class == class_id)[0][:10]
        for i, idx in enumerate(members):
            ax_curves.plot(run.features[idx][:run.n_radial], color=cluster_colour(class_id), alpha=0.4, label=name if i == 0 else None)
    style_axis(ax_curves, 'radial profile feature, up to 10 particles per class', xlabel='radial bin', ylabel='mean intensity', legend=True)


# plot_pca: PCA scree plot next to the embedding coloured by true class
def plot_pca(run, axes):
    ax_scree, ax_scatter = axes
    ratio = run.pca.explained_variance_ratio_
    x = np.arange(1, len(ratio) + 1)
    ax_scree.bar(x, ratio, color=CLUSTER_COLOURS[0])
    ax_scree.plot(x, np.cumsum(ratio), '-o', color=CLUSTER_COLOURS[1], ms=3, label='cumulative')
    style_axis(ax_scree, 'PCA explained variance', xlabel='component', ylabel='variance ratio', legend=True)

    groups = [(run.true_class == class_id, cluster_colour(class_id), name) for class_id, name in enumerate(run.names)]
    scatter_groups(ax_scatter, run.embedding, groups, 'embedding (PC1 vs PC2), coloured by true class', size=22)


# plot_kmeans: embedding coloured by KMeans label
def plot_kmeans(run, ax):
    scatter_labels(ax, run.embedding, run.kmeans_labels, f'KMeans (k={len(set(run.kmeans_labels))}) on the embedding')


# plot_hdbscan: embedding coloured by HDBSCAN label, with cluster / noise counts
def plot_hdbscan(run, ax):
    n_noise = int((run.hdbscan_labels == -1).sum())
    n_clusters = len(set(run.hdbscan_labels) - {-1})
    scatter_labels(ax, run.embedding, run.hdbscan_labels, f'HDBSCAN: {n_clusters} clusters, {n_noise} noise')


# plot_composition: stacked bar of the true-species make-up of each KMeans cluster
def plot_composition(run, ax):
    clusters = sorted(set(run.kmeans_labels))
    bottom = np.zeros(len(clusters))
    for class_id, name in enumerate(run.names):
        heights = [int(np.sum((run.kmeans_labels == c) & (run.true_class == class_id))) for c in clusters]
        ax.bar(clusters, heights, bottom=bottom, label=name, color=cluster_colour(class_id))
        bottom += heights
    ax.set_xticks(clusters)
    style_axis(ax, 'cluster composition by true species', xlabel='KMeans cluster', ylabel='particles', legend=True)


# plot_class_averages: row of class-average thumbnails with member counts
def plot_class_averages(run, axes):
    class_average_grid(axes, run.w.cluster_ids, run.w.averages, lambda c: f'cluster {c} (n={run.w.class_counts[c]})')

# --- drivers ---
# save_panels: write each stage panel as a numbered PNG
def save_panels(run, outdir):
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_subvolumes(run, ax)
    finish(fig, outdir / '01_subvolumes.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_features(run, axes)
    finish(fig, outdir / '02_features.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_pca(run, axes)
    finish(fig, outdir / '03_pca.png')

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_kmeans(run, ax)
    finish(fig, outdir / '04_kmeans.png')

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_hdbscan(run, ax)
    finish(fig, outdir / '05_hdbscan.png')

    fig, ax = plt.subplots(figsize=(5.5, 4))
    plot_composition(run, ax)
    finish(fig, outdir / '06_composition.png')

    n = max(2, len(run.w.cluster_ids))
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))
    plot_class_averages(run, np.atleast_1d(axes))
    finish(fig, outdir / '07_class_averages.png')

# save_storyboard: lay every classify panel out on one 00_classify_storyboard.png figure
def save_storyboard(run, outdir):
    fig = plt.figure(figsize=(19, 15))
    grid = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    plot_subvolumes(run, fig.add_subplot(grid[0, 0]))
    plot_features(run, [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])])
    plot_pca(run, [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])])
    plot_composition(run, fig.add_subplot(grid[1, 2]))
    plot_kmeans(run, fig.add_subplot(grid[2, 0]))
    plot_hdbscan(run, fig.add_subplot(grid[2, 1]))

    n = max(2, len(run.w.cluster_ids))
    plot_class_averages(run, subgrid_axes(fig, grid[2, 2], n))

    finish(fig, outdir / '00_classify_storyboard.png')

# render: run the mocked classify stage and write its figures, returning the output directory
def render(config: DiagramConfig) -> Path:
    return render_diagram(config, ClassifyRun, save_panels, save_storyboard)
