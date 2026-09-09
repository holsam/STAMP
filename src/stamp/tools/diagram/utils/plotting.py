'''
STAMP: shared matplotlib helpers
'''

# Import external dependencies
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import internal STAMP objects
from stamp.tools.diagram.utils.identify import read_atom_coordinates

# --- palette ---
GREY = 'gray'
BLUE = '#4da6ff'
ORANGE = '#ff9d4d'
GREEN = '#5dff8f'
RED = '#ff5d5d'
# CLUSTER_COLOURS: per-cluster colour cycle
CLUSTER_COLOURS = ('#4da6ff', '#ff9d4d', '#7bd88f', '#c77dff', '#ffd166', '#ff8fab', '#4dd0e1')
NOISE_COLOUR = '#b0b0b0'

# _PROTEIN_CMAP: single-colour colormap for the protein-density overlay
_PROTEIN_CMAP = matplotlib.colors.ListedColormap([ORANGE])

# cluster_colour: colour for a cluster index, wrapping round CLUSTER_COLOURS
def cluster_colour(index: int) -> str:
    return CLUSTER_COLOURS[index % len(CLUSTER_COLOURS)]

# --- low-level axis helpers ---
# bare: strip x and y ticks from an axis
def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])

# frame: square image-space axis with y running downward and no ticks
def frame(ax, field_px, title):
    ax.set_xlim(0, field_px)
    ax.set_ylim(field_px, 0)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    bare(ax)

# style_axis: title / label / legend / tick-size boilerplate
def style_axis(ax, title=None, xlabel=None, ylabel=None, *, legend=False, title_size=9, label_size=8, tick_size=6):
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size)
    if legend:
        ax.legend(fontsize=tick_size)
    ax.tick_params(labelsize=tick_size)

# finish: tight-layout, save at 200 dpi, close the figure and log the path
def finish(fig, path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {path}')

# subgrid_axes: a 1xN row of axes inside one storyboard grid cell
def subgrid_axes(fig, cell, n, *, wspace=0.2, hspace=None):
    inner = cell.subgridspec(1, n, wspace=wspace, hspace=hspace)
    return [fig.add_subplot(inner[i]) for i in range(n)]

# --- structure / particle imagery ---
# structure_projection_scatter: the actual atoms of a PDB as a depth-coloured xy scatter, on the same grid as its simulated density map
def structure_projection_scatter(ax, pdb_path, box, pixel_size, cmap='viridis'):
    coordinates = read_atom_coordinates(pdb_path)
    centred = coordinates - coordinates.mean(axis=0)
    xy = centred[:, :2] / pixel_size + box / 2.0
    ax.scatter(xy[:, 0], xy[:, 1], c=centred[:, 2], cmap=cmap, s=2, linewidths=0)
    ax.set_xlim(0, box)
    ax.set_ylim(box, 0)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    bare(ax)

# tile_particles: lay a list of box-sized particles into one grid canvas, gaps left as NaN
def tile_particles(particles, box, columns=None):
    n = len(particles)
    columns = columns or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / columns))
    canvas = np.full((rows * box, columns * box), np.nan)
    for index, particle in enumerate(particles):
        r, c = divmod(index, columns)
        canvas[r * box:(r + 1) * box, c * box:(c + 1) * box] = particle
    return canvas

# draw_vesicles: outline each vesicle as an unfilled circle
def draw_vesicles(ax, vesicles, colour='w', lw=0.8):
    for v in vesicles:
        ax.add_patch(plt.Circle((v['cx'], v['cy']), v['r'], fill=False, edgecolor=colour, lw=lw, alpha=0.5))

# show_segmentation: membrane in grey, protein density in solid orange, optional picks
def show_segmentation(ax, segmentation, field_px, picks=None):
    ax.imshow(segmentation == 1, cmap='Greys', alpha=0.55, vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(segmentation != 2, segmentation), cmap=_PROTEIN_CMAP, alpha=0.85)
    if picks is not None and len(picks):
        ax.scatter(picks[:, 0], picks[:, 1], s=14, c=GREEN, label='consensus picks')
    ax.set_xlim(0, field_px)
    ax.set_ylim(field_px, 0)
    bare(ax)

# --- shared panels ---
# class_average_grid: row of class-average thumbnails (title_fn(cluster_id) -> str titles each), leftover axes switched off
def class_average_grid(axes, ids, averages, title_fn, *, title_size=8):
    for ax, cluster_id in zip(axes, ids):
        ax.imshow(averages[cluster_id], cmap=GREY)
        ax.set_title(title_fn(cluster_id), fontsize=title_size)
        bare(ax)
    for ax in axes[len(ids):]:
        ax.set_axis_off()

# decoy_hist: real-vs-decoy fit-CC histogram with a PASS / FAIL title
def decoy_hist(ax, real_scores, decoy_scores, passed):
    bins = np.linspace(min(real_scores.min(), decoy_scores.min()), max(real_scores.max(), decoy_scores.max()), 16)
    ax.hist(decoy_scores, bins=bins, alpha=0.6, color='#b0b0b0', label='decoy')
    ax.hist(real_scores, bins=bins, alpha=0.7, color=BLUE, label='real')
    style_axis(ax, f'decoy control: {"PASS" if passed else "FAIL"}', xlabel='fit CC', legend=True)

# score_heatmap: class x candidate cross-correlation heatmap with per-cell values; returns (image, matrix) so the caller can tweak labels / colorbar
def score_heatmap(ax, cluster_ids, names, score_matrix, *, row_labels=None, title=None, colorbar=True):
    matrix = np.array([[score_matrix[c][n] for n in names] for c in cluster_ids])
    image = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=15, ha='right')
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels(row_labels or [f'class {c}' for c in cluster_ids], fontsize=7)
    for (r, c), value in np.ndenumerate(matrix):
        ax.text(c, r, f'{value:.2f}', ha='center', va='center', fontsize=7, color='w' if value < 0.6 else 'k')
    if title is not None:
        ax.set_title(title, fontsize=9)
    if colorbar:
        image.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return image, matrix

# style_embedding: style_axis preset for a PC1 / PC2 embedding scatter
def style_embedding(ax, title):
    style_axis(ax, title, xlabel='PC1', ylabel='PC2', legend=True)

# scatter_groups: scatter an embedding as labelled groups of (mask, colour, label)
def scatter_groups(ax, embedding, groups, title, *, size=24):
    for mask, colour, label in groups:
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=size, c=colour, edgecolors='k', linewidths=0.3, label=label)
    style_embedding(ax, title)

# scatter_labels: scatter an embedding coloured by integer cluster label (-1 = noise)
def scatter_labels(ax, embedding, labels, title, *, size=24):
    for label in sorted(set(labels)):
        sel = labels == label
        if label == -1:
            ax.scatter(embedding[sel, 0], embedding[sel, 1], s=16, c=NOISE_COLOUR, marker='x', label='noise')
        else:
            ax.scatter(embedding[sel, 0], embedding[sel, 1], s=size, c=cluster_colour(label), edgecolors='k', linewidths=0.3, label=f'cluster {label}')
    style_embedding(ax, title)
