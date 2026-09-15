'''
STAMP: mock-only plot helpers
'''

# Import external dependencies
import matplotlib.pyplot as plt

# Import internal STAMP objects
from stamp.tools.diagram.utils.identify import read_atom_coordinates
from stamp.utils.plotting.core import bare

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

# draw_vesicles: outline each vesicle as an unfilled circle
def draw_vesicles(ax, vesicles, colour='w', lw=0.8):
    for v in vesicles:
        ax.add_patch(plt.Circle((v['cx'], v['cy']), v['r'], fill=False, edgecolor=colour, lw=lw, alpha=0.5))
