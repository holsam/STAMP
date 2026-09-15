'''
STAMP: identify-stage plots
'''

# Import external dependencies
import numpy as np

# Import internal STAMP objects
from stamp.utils.plotting.core import BLUE, style_axis

# decoy_hist: real-vs-decoy fit-CC histogram
def decoy_hist(ax, real_scores, decoy_scores, passed):
    real_scores, decoy_scores = np.asarray(real_scores, dtype=float), np.asarray(decoy_scores, dtype=float)
    bins = np.linspace(min(real_scores.min(), decoy_scores.min()), max(real_scores.max(), decoy_scores.max()), 16)
    ax.hist(decoy_scores, bins=bins, alpha=0.6, color='#b0b0b0', label='decoy')
    ax.hist(real_scores, bins=bins, alpha=0.7, color=BLUE, label='real')
    style_axis(ax, f'decoy control: {"PASS" if passed else "FAIL"}', xlabel='fit CC', legend=True)

# score_heatmap: class x candidate cross-correlation heatmap with per-cell values
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
