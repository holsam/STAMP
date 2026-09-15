'''
STAMP: classify-stage plots
'''

# Import external dependencies
import numpy as np

# Import internal STAMP objects
from stamp.utils.plotting.core import GREY, NOISE_COLOUR, bare, cluster_colour, style_embedding

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
