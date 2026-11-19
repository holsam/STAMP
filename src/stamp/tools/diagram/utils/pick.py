'''
STAMP: 2D mocked functions for pick
'''

# Import external dependencies
import numpy as np

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior
from stamp.tools.diagram.utils.carriers import Carrier, nearest_surface
from stamp.tools.diagram.utils.geometry import UnionFind

# simulate_picks_file: one external picker's output, found particles (with recall and position jitter), contaminants both pickers fire on, and private false picks anywhere in the field
def simulate_picks_file(true_xy: np.ndarray, confounders: np.ndarray, field_px: int, rng: np.random.Generator, *, recall: float, jitter_px: float, n_false: int, confounder_recall: float = 0.8) -> np.ndarray:
    found = true_xy[rng.random(len(true_xy)) < recall]
    shared = confounders[rng.random(len(confounders)) < confounder_recall]
    private = rng.uniform(0.0, field_px, size=(n_false, 2))
    points = np.vstack([found, shared, private])
    return points + rng.normal(0.0, jitter_px, size=points.shape)

# infer_orientations: orientation angle per pick (nan when unknown) and whether the pick is usable
def infer_orientations(prior: OrientationPrior, carriers: list[Carrier], points: np.ndarray, rng: np.random.Generator, max_surface_distance_px: float) -> tuple[np.ndarray, np.ndarray]:
    if prior is OrientationPrior.NONE:  # orientations are ignored
        return np.full(len(points), np.nan), np.ones(len(points), dtype=bool)
    distance, angle, _owner = nearest_surface(carriers, points)
    if prior is OrientationPrior.AXIS:  # the picks file carries the filament direction, with some error
        return angle + rng.normal(0.0, np.deg2rad(5.0), size=len(points)), np.ones(len(points), dtype=bool)
    oriented = distance <= max_surface_distance_px     # normal: picks far from any membrane keep no orientation and classify skips them
    return np.where(oriented, angle, np.nan), oriented

# beam_deviation: angle (degrees) between each normal and the beam-orthogonal plane
def beam_deviation(n: int, rng: np.random.Generator) -> np.ndarray:
    return 90.0 * rng.beta(1.3, 3.0, size=n)

# circular_mean: mean of angles in radians, nan when every angle is nan
def circular_mean(angles: np.ndarray) -> float:
    valid = angles[~np.isnan(angles)]
    return float(np.angle(np.exp(1j * valid).mean())) if len(valid) else float('nan')

# reconcile: union-find grouping of pickers' points, keeping groups both pickers contributed to (STAMP's intersection consensus)
def reconcile(points_a: np.ndarray, points_b: np.ndarray, distance_px: float) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), []
    allp = np.vstack([points_a, points_b])
    picker = np.array([0] * len(points_a) + [1] * len(points_b))
    uf = UnionFind(len(allp))
    for i in range(len(allp)):
        for j in np.where(np.linalg.norm(allp - allp[i], axis=1) < distance_px)[0]:
            uf.union(i, int(j))
    groups: dict[int, list[int]] = {}
    for i in range(len(allp)):
        groups.setdefault(uf.find(i), []).append(i)
    kept, rejected, members = [], [], []
    for group in groups.values():
        centre = allp[group].mean(axis=0)
        if len(set(picker[group])) >= 2:
            kept.append(centre)
            members.append(np.array(group))
        else:
            rejected.append(centre)
    return (np.array(kept) if kept else np.empty((0, 2)), np.array(rejected) if rejected else np.empty((0, 2)), members)
