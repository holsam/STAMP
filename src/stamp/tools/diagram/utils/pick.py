'''
STAMP: 2D mocked functions for pick
'''

# Import external dependencies
import numpy as np

# Import internal STAMP objects
from stamp.tools.diagram.utils.geometry import UnionFind

# sample_membrane_surface: points walked around each vesicle membrane, as a native picker would score surface positions
def sample_membrane_surface(vesicles, rng, spacing_deg=5.0, jitter_px=1.5):
    points, owner = [], []
    for index, v in enumerate(vesicles):
        for angle in np.arange(0.0, 360.0, spacing_deg):
            theta = np.deg2rad(angle)
            radius = v['r'] + rng.normal(0.0, jitter_px)
            points.append((v['cx'] + radius * np.cos(theta), v['cy'] + radius * np.sin(theta)))
            owner.append(index)
    return np.array(points), np.array(owner)

# score_surface: high where a surface point sits on a real particle, low elsewhere, plus per-evaluation noise (a stand-in for a density / template score)
def score_surface(points, true_xy, capture_px, rng, noise=0.28):
    if len(true_xy) == 0:
        nearest = np.full(len(points), np.inf)
    else:
        nearest = np.linalg.norm(points[:, None, :] - true_xy[None, :, :], axis=2).min(axis=1)
    base = np.where(nearest < capture_px, 1.0, 0.12)
    return base + rng.normal(0.0, noise, size=len(points)), nearest

# reconcile: union-find grouping of two pickers' points, keeping only groups both pickers contributed to (STAMP's intersection consensus)
def reconcile(points_a, points_b, distance_px):
    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    allp = np.vstack([points_a, points_b])
    picker = np.array([0] * len(points_a) + [1] * len(points_b))
    uf = UnionFind(len(allp))
    for i in range(len(allp)):
        near = np.where(np.linalg.norm(allp - allp[i], axis=1) < distance_px)[0]
        for j in near:
            uf.union(i, int(j))
    groups: dict[int, list[int]] = {}
    for i in range(len(allp)):
        groups.setdefault(uf.find(i), []).append(i)
    kept, rejected = [], []
    for members in groups.values():
        centre = allp[members].mean(axis=0)
        if len(set(picker[members])) >= 2:
            kept.append(centre)
        else:
            rejected.append(centre)
    return (np.array(kept) if kept else np.empty((0, 2)), np.array(rejected) if rejected else np.empty((0, 2)))
