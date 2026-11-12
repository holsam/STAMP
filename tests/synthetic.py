'''
STAMP: synthetic particles with known orientations, shifts and wedges
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.ndimage import gaussian_filter, shift as shift_volume
from scipy.spatial.transform import Rotation

# Import internal STAMP objects
from stamp.classify.frames import azimuth_matrix, flip_matrix, rotate_density
from stamp.classify.store import ParticleStore
from stamp.classify.wedge import apply_wedge, box_wedge
from stamp.picking.geometry import quaternion_from_reference_to
from stamp.schemas.particles import OrientationPrior
from stamp.schemas.subvolumes import Subvolumes

# asymmetric_object: smooth volume of random gaussian blobs, no symmetry, so every rotation and flip is distinguishable
def asymmetric_object(box: int, seed: int, *, n_blobs: int = 7, spread: float = 0.22) -> np.ndarray:
    rng = np.random.default_rng(seed)
    volume = np.zeros((box,) * 3, dtype=np.float32)
    centre = (box - 1) / 2.0
    for _ in range(n_blobs):
        index = np.round(centre + rng.normal(0.0, spread * box, 3)).astype(int).clip(2, box - 3)
        volume[tuple(index)] += rng.uniform(0.5, 1.5)
    smooth = gaussian_filter(volume, box / 14.0)
    return smooth / smooth.max()

# rod_object: filament along z with a repeating asymmetric pattern, polarity and axial position are both observable
def rod_object(box: int, seed: int, *, repeat: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centre = (box - 1) / 2.0
    axis = np.arange(box) - centre
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    body = np.exp(-(x**2 + y**2) / (2 * (box / 10.0) ** 2))
    phase = rng.uniform(0, 2 * np.pi)
    lump = np.exp(-((x - box / 7.0 * np.cos(2 * np.pi * z / repeat + phase)) ** 2 + (y - box / 7.0 * np.sin(2 * np.pi * z / repeat + phase)) ** 2) / (2 * (box / 16.0) ** 2))
    cap = 2.0 * np.exp(-((x - box / 5.0) ** 2 + y**2 + (z - box / 4.0) ** 2) / (2 * (box / 12.0) ** 2))     # off-axis cap at one end breaks the up-down symmetry so polarity is observable
    return (body + lump + cap).astype(np.float32)

# make_particle: the object moved by rotation then shift, limited to the box-frame wedge, plus white noise
def make_particle(
    obj: np.ndarray,
    rotation: np.ndarray,
    shift: tuple[float, float, float],
    wedge: np.ndarray | None,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    moved = shift_volume(rotate_density(obj, rotation), shift, order=1)
    if wedge is not None:
        moved = apply_wedge(moved, wedge)
    return (moved + rng.normal(0.0, noise_sigma, obj.shape)).astype(np.float32)

# SyntheticSet: particles with known class, pose and shift, as a prepared store
@dataclass(frozen=True)
class SyntheticSet:
    store: ParticleStore
    labels: np.ndarray
    rotations: np.ndarray
    shifts: np.ndarray

# synthetic_set: n_per_class particles of each object, poses drawn for the prior, one cache per fake tomogram, wedges from random orientations
def synthetic_set(
    directory: Path,
    objects: list[np.ndarray],
    prior: OrientationPrior,
    n_per_class: int,
    *,
    noise_sigma: float = 0.1,
    tilt_max: float | None = 60.0,
    seed: int = 0,
    max_shift: int = 2,
) -> SyntheticSet:
    rng = np.random.default_rng(seed)
    box = objects[0].shape[0]
    labels = np.repeat(np.arange(len(objects)), n_per_class)
    rng.shuffle(labels)
    n = len(labels)
    if prior is OrientationPrior.NONE:
        rotations = Rotation.random(n, random_state=seed).as_matrix()
        orientations = [None] * n
    else:
        flips = [flip_matrix() if prior is OrientationPrior.AXIS and rng.random() < 0.5 else np.eye(3) for _ in range(n)]
        rotations = np.stack([azimuth_matrix(a) @ f for a, f in zip(rng.uniform(0, 360, n), flips)])
        orientations = [quaternion_from_reference_to(rng.normal(size=3)) for _ in range(n)]
    shifts = rng.integers(-max_shift, max_shift + 1, (n, 3)) * (np.array([0, 0, 2]) if prior is OrientationPrior.AXIS else 1)
    wedges = [box_wedge(box, o, tilt_max, 90.0) for o in orientations]
    volumes = np.stack([make_particle(objects[label], rotation, tuple(shift), wedge, noise_sigma, rng) for label, rotation, shift, wedge in zip(labels, rotations, shifts, wedges)])
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    for tomogram in range(3):
        rows = np.arange(tomogram, n, 3)
        np.save(directory / f't{tomogram}.npy', volumes[rows])
        index += [(f't{tomogram}', row) for row in range(len(rows))]
    order = np.concatenate([np.arange(t, n, 3) for t in range(3)])     # store index i holds particle order[i]
    store = ParticleStore(Subvolumes(directory, index, box), [orientations[i] for i in order], tilt_max, 90.0)
    return SyntheticSet(store, labels[order], rotations[order], shifts[order])
