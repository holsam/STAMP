'''
STAMP: mock carriers (vesicles, filaments) that particles sit on
'''

# Import external dependencies
import numpy as np
from scipy.spatial import cKDTree

# Import internal STAMP objects
from stamp.schemas.particles import OrientationPrior

# Carrier: a vesicle cross-section ('circle', cx, cy, r) or a straight filament ('line', x0, y0, x1, y1)
Carrier = dict

# make_vesicles: non-overlapping circular vesicle cross-sections
def make_vesicles(field_px: int, n_vesicles: int, rng: np.random.Generator, radius_px: tuple[float, float] = (52, 104), margin: float = 20) -> list[Carrier]:
    vesicles: list[Carrier] = []
    attempts = 0
    while len(vesicles) < n_vesicles and attempts < 400:
        attempts += 1
        r = float(rng.uniform(*radius_px))
        cx = float(rng.uniform(r + margin, field_px - r - margin))
        cy = float(rng.uniform(r + margin, field_px - r - margin))
        if all(np.hypot(cx - v['cx'], cy - v['cy']) > r + v['r'] + margin for v in vesicles):
            vesicles.append({'kind': 'circle', 'cx': cx, 'cy': cy, 'r': r})
    return vesicles

# make_filaments: straight filaments with random direction, kept clear of each other
def make_filaments(field_px: int, n_filaments: int, rng: np.random.Generator, length_px: tuple[float, float] = (260, 460), margin: float = 40, gap: float = 60) -> list[Carrier]:
    filaments: list[Carrier] = []
    attempts = 0
    while len(filaments) < n_filaments and attempts < 800:
        attempts += 1
        length = float(rng.uniform(*length_px))
        angle = float(rng.uniform(0, np.pi))
        centre = rng.uniform(margin + length / 2, field_px - margin - length / 2, size=2)
        half = 0.5 * length * np.array([np.cos(angle), np.sin(angle)])
        candidate = {'kind': 'line', 'x0': float(centre[0] - half[0]), 'y0': float(centre[1] - half[1]), 'x1': float(centre[0] + half[0]), 'y1': float(centre[1] + half[1])}
        inside = all(margin <= v <= field_px - margin for v in (candidate['x0'], candidate['y0'], candidate['x1'], candidate['y1']))
        if inside and all(carrier_distance(f, *_sample_line(candidate, 8.0)[0].T).min() > gap for f in filaments):
            filaments.append(candidate)
    return filaments

# make_carriers: what the picks sit on for each prior
def make_carriers(prior: OrientationPrior, field_px: int, n_vesicles: int, n_filaments: int, rng: np.random.Generator) -> list[Carrier]:
    if prior is OrientationPrior.NORMAL:
        return make_vesicles(field_px, n_vesicles, rng)
    if prior is OrientationPrior.AXIS:
        return make_filaments(field_px, n_filaments, rng)
    return []

# carrier_distance: distance from points to one carrier's centre line
def carrier_distance(carrier: Carrier, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if carrier['kind'] == 'circle':
        return np.abs(np.hypot(x - carrier['cx'], y - carrier['cy']) - carrier['r'])
    start, end = np.array([carrier['x0'], carrier['y0']]), np.array([carrier['x1'], carrier['y1']])
    along = end - start
    t = np.clip(((x - start[0]) * along[0] + (y - start[1]) * along[1]) / (along @ along), 0.0, 1.0)
    return np.hypot(x - (start[0] + t * along[0]), y - (start[1] + t * along[1]))

# _sample_line: evenly spaced points along a filament and filament direction at each
def _sample_line(carrier: Carrier, spacing_px: float) -> tuple[np.ndarray, np.ndarray]:
    start, end = np.array([carrier['x0'], carrier['y0']]), np.array([carrier['x1'], carrier['y1']])
    n = max(2, int(np.linalg.norm(end - start) / spacing_px))
    points = start + np.linspace(0.0, 1.0, n)[:, None] * (end - start)
    return points, np.full(n, np.arctan2(end[1] - start[1], end[0] - start[0]))

# surface_points: points along a carrier and orientation angle each would be picked with (outward normal on a vesicle, filament direction on a filament, in radians)
def surface_points(carrier: Carrier, spacing_px: float) -> tuple[np.ndarray, np.ndarray]:
    if carrier['kind'] == 'line':
        return _sample_line(carrier, spacing_px)
    theta = np.arange(0.0, 2 * np.pi, spacing_px / carrier['r'])
    return np.c_[carrier['cx'] + carrier['r'] * np.cos(theta), carrier['cy'] + carrier['r'] * np.sin(theta)], theta

# place_on: a position on a carrier with a small jitter off centre line, and orientation angle there
def place_on(carrier: Carrier, rng: np.random.Generator, jitter_px: float = 1.5) -> tuple[float, float, float]:
    if carrier['kind'] == 'circle':
        theta = float(rng.uniform(0, 2 * np.pi))
        radius = carrier['r'] + float(rng.normal(0.0, jitter_px))
        return carrier['cx'] + radius * np.cos(theta), carrier['cy'] + radius * np.sin(theta), theta
    t = float(rng.uniform(0.05, 0.95))
    x = carrier['x0'] + t * (carrier['x1'] - carrier['x0'])
    y = carrier['y0'] + t * (carrier['y1'] - carrier['y0'])
    angle = float(np.arctan2(carrier['y1'] - carrier['y0'], carrier['x1'] - carrier['x0']))
    off = float(rng.normal(0.0, jitter_px))
    return x - off * np.sin(angle), y + off * np.cos(angle), angle

# nearest_surface: for each point, distance to nearest carrier surface sample, orientation angle there and owning carrier
def nearest_surface(carriers: list[Carrier], points: np.ndarray, spacing_px: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = [surface_points(carrier, spacing_px) for carrier in carriers]
    surface = np.vstack([s[0] for s in samples])
    angles = np.concatenate([s[1] for s in samples])
    owner = np.concatenate([np.full(len(s[0]), i) for i, s in enumerate(samples)])
    distance, index = cKDTree(surface).query(points)
    return distance, angles[index], owner[index]

# carrier_of: index of carrier nearest each point
def carrier_of(carriers: list[Carrier], points: np.ndarray) -> np.ndarray:
    return np.argmin([carrier_distance(c, points[:, 0], points[:, 1]) for c in carriers], axis=0)
