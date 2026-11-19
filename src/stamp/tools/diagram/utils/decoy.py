'''
STAMP: 2D mocked functions for decoy generation
'''

# Import external dependencies
import numpy as np
from scipy.ndimage import gaussian_filter

# Import internal STAMP objects
from stamp.tools.diagram.utils.carriers import Carrier, carrier_distance

# _clear_of: True where a point is further than radius from every row of others
def _clear_of(point: np.ndarray, others: np.ndarray, radius: float) -> bool:
    return len(others) == 0 or bool(np.linalg.norm(others - point, axis=1).min() > radius)

# shifted_decoys: displace each real pick by a large random vector, away from carrier surface and from every real pick
def shifted_decoys(picks: np.ndarray, carriers: list[Carrier], true_xy: np.ndarray, field_px: int, box: int, rng: np.random.Generator, *, min_shift: float = 1.6, max_shift: float = 3.2, tries: int = 30) -> np.ndarray:
    decoys = []
    for pick in picks:
        for _ in range(tries):
            angle, length = rng.uniform(0, 2 * np.pi), rng.uniform(min_shift, max_shift) * box
            candidate = pick + length * np.array([np.cos(angle), np.sin(angle)])
            inside = np.all((candidate > box) & (candidate < field_px - box))
            off_surface = all(carrier_distance(c, candidate[0], candidate[1]) > 0.5 * box for c in carriers)
            if inside and off_surface and _clear_of(candidate, true_xy, 0.8 * box) and _clear_of(candidate, np.array(decoys), 0.5 * box):
                decoys.append(candidate)
                break
    return np.array(decoys) if decoys else np.empty((0, 2))

# random_volume_decoys: one decoy per real pick at a uniformly random position clear of every real pick and of border
def random_volume_decoys(n: int, true_xy: np.ndarray, field_px: int, box: int, rng: np.random.Generator, *, tries: int = 200) -> np.ndarray:
    decoys = []
    for _ in range(n * tries):
        candidate = rng.uniform(box, field_px - box, size=2)
        if _clear_of(candidate, true_xy, 0.8 * box) and _clear_of(candidate, np.array(decoys), 0.5 * box):
            decoys.append(candidate)
        if len(decoys) == n:
            break
    return np.array(decoys) if decoys else np.empty((0, 2))

# synthetic_noise_tile: a Gaussian-noise volume slice with a faint membrane-like shell and decoy positions sampled on shell
def synthetic_noise_tile(rng: np.random.Generator, n: int, size_px: int = 240) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size_px, 0:size_px]
    radius = 0.34 * size_px
    shell = np.exp(-((np.hypot(xx - size_px / 2, yy - size_px / 2) - radius) ** 2) / (2 * 2.6 ** 2))
    tile = gaussian_filter(rng.normal(0.0, 1.0, size=(size_px, size_px)), 1.2) + 1.1 * shell
    theta = rng.uniform(0, 2 * np.pi, size=n)
    return tile.astype(np.float32), np.c_[size_px / 2 + radius * np.cos(theta), size_px / 2 + radius * np.sin(theta)]
