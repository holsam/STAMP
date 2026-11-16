'''
STAMP: classify-stage plots
'''

# Import external dependencies
import matplotlib.pyplot as plt, numpy as np
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.cluster import ClusteringResult
from stamp.utils.options.literals import PlotFormat
from stamp.utils.plotting.core import GREY, NOISE_COLOUR, bare, central_slice, cluster_colour, finish, plot_path, style_embedding

# class_average_grid: row of class-average thumbnails
def class_average_grid(axes, ids, averages, title_fn, *, title_size=8):
    for ax, cluster_id in zip(axes, ids):
        ax.imshow(averages[cluster_id], cmap=GREY)
        ax.set_title(title_fn(cluster_id), fontsize=title_size)
        bare(ax)
    for ax in axes[len(ids):]:
        ax.set_axis_off()

# scatter_labels: scatter an embedding coloured by integer cluster label (-1 = noise)
def scatter_labels(ax, embedding, labels, title, *, size=24):
    labels = np.asarray(labels)
    for label in sorted(set(labels)):
        sel = labels == label
        if label == -1:
            ax.scatter(embedding[sel, 0], embedding[sel, 1], s=16, c=NOISE_COLOUR, marker='x', label='noise')
        else:
            ax.scatter(embedding[sel, 0], embedding[sel, 1], s=size, c=cluster_colour(label), edgecolors='k', linewidths=0.3, label=f'cluster {label}')
    style_embedding(ax, title)

# plot_classify: create plots (PCA embedding scatter + class-average thumbnail grid) for one orientation prior
def plot_classify(result_by_half: dict[str, ClusteringResult], averages_by_half: dict[str, dict[str, tuple[np.ndarray, int]]], output_dir: Path, fmt: PlotFormat, tag: str) -> None:
    halves = sorted(result_by_half)
    fig, axes = plt.subplots(1, len(halves), figsize=(5.5 * len(halves), 4), squeeze=False)
    for ax, half in zip(axes[0], halves):
        result = result_by_half[half]
        scatter_labels(ax, result.embedding[:, :2], result.labels, f'{tag} half {half}: PCA -> {result.labels.max() + 1} cluster(s)')
    finish(fig, plot_path(output_dir, f'embedding_{tag}', fmt))

    for half, averages in averages_by_half.items():
        ids = sorted(averages)
        if not ids:
            continue
        fig, axes = plt.subplots(1, max(2, len(ids)), figsize=(2.6 * max(2, len(ids)), 3), squeeze=False)
        class_average_grid(axes[0], ids, {cid: central_slice(volume) for cid, (volume, _count) in averages.items()}, lambda c: c)
        finish(fig, plot_path(output_dir, f'class_averages_{tag}_half{half}', fmt))
