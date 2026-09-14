'''
STAMP: generic matplotlib helpers
'''

# Import external dependencies
import numpy as np, matplotlib, matplotlib.pyplot as plt, warnings
from pathlib import Path
from typing import Literal

# Import internal STAMP objects
from stamp.utils.log import log

matplotlib.use('Agg')

# --- palette ---
GREY = 'gray'
BLUE = '#4da6ff'
ORANGE = '#ff9d4d'
GREEN = '#5dff8f'
RED = '#ff5d5d'
CLUSTER_COLOURS = ('#4da6ff', '#ff9d4d', '#7bd88f', '#c77dff', '#ffd166', '#ff8fab', '#4dd0e1')
NOISE_COLOUR = '#b0b0b0'
PROTEIN_CMAP = matplotlib.colors.ListedColormap([ORANGE])

# PlotFormat: image formats a plot can be saved as
PlotFormat = Literal['png', 'jpg', 'tiff', 'svg']

# cluster_colour: colour for a cluster index, wrapping round CLUSTER_COLOURS
def cluster_colour(index: int) -> str:
    return CLUSTER_COLOURS[index % len(CLUSTER_COLOURS)]

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

# style_axis: title/label/legend/tick-size style
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

# plot_path: return the filename/path for a given plot
def plot_path(directory: Path, stem: str, fmt: PlotFormat) -> Path:
    return directory / f'{stem}.{fmt}'

# finish: save plot to path and close figure
def finish(fig, path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log.info(f'Wrote plot: {path}')

# subgrid_axes: a 1xN row of axes inside one storyboard grid cell
def subgrid_axes(fig, cell, n, *, wspace=0.2, hspace=None):
    inner = cell.subgridspec(1, n, wspace=wspace, hspace=hspace)
    return [fig.add_subplot(inner[i]) for i in range(n)]

# central_slice: the mid-Z 2D plane of a 3D volume, for imshow
def central_slice(volume: np.ndarray) -> np.ndarray:
    return volume[volume.shape[0] // 2]

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

# show_segmentation: membrane in grey, protein density in solid orange, optional picks
def show_segmentation(ax, segmentation, field_px, picks=None):
    ax.imshow(segmentation == 1, cmap='Greys', alpha=0.55, vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(segmentation != 2, segmentation), cmap=PROTEIN_CMAP, alpha=0.85)
    if picks is not None and len(picks):
        ax.scatter(picks[:, 0], picks[:, 1], s=14, c=GREEN, label='consensus picks')
    ax.set_xlim(0, field_px)
    ax.set_ylim(field_px, 0)
    bare(ax)

# style_embedding: style_axis preset for a PC1 / PC2 embedding scatter
def style_embedding(ax, title):
    style_axis(ax, title, xlabel='PC1', ylabel='PC2', legend=True)

# scatter_groups: scatter embedding as labelled groups of (mask, colour, label)
def scatter_groups(ax, embedding, groups, title, *, size=24):
    for mask, colour, label in groups:
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=size, c=colour, edgecolors='k', linewidths=0.3, label=label)
    style_embedding(ax, title)
