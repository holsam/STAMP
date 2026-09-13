'''
STAMP: predicted structure to density map
'''

# Import external dependencies
import numpy as np
from itertools import product
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate

# _ELEMENT_WEIGHT: approximate scattering weight per element; unknown non-blank elements fall back to carbon
_ELEMENT_WEIGHT = { 'H': 1.0, 'C': 6.0, 'N': 7.0, 'O': 8.0, 'F': 9.0, 'P': 15.0, 'S': 16.0, 'CL': 17.0, 'FE': 26.0, 'ZN': 30.0, 'MG': 12.0, 'MN': 25.0, 'CA': 20.0}
_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493  # 2*sqrt(2*ln2)

# read_pdb_atoms: (N, 3) coordinates in Angstrom, (N,) weights and (N,) element symbols
def read_pdb_atoms(structure_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    coordinates: list[tuple[float, float, float]] = []
    weights: list[float] = []
    elements: list[str] = []
    for line in structure_path.read_text().splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        columns_element = line[76:78].strip().upper()
        name_element = line[12:16].strip()[:1].upper()
        element = columns_element or name_element or 'H'
        weights.append(_ELEMENT_WEIGHT.get(element, 6.0))
        elements.append(element)
    if not coordinates:
        raise ValueError(f'{structure_path!r} has no ATOM/HETATM records')
    return np.array(coordinates), np.array(weights), elements

# _splat: trilinear deposition of per-atom weight into the volume
def _splat(voxel_xyz: np.ndarray, weights: np.ndarray, box_voxels: int) -> np.ndarray:
    volume = np.zeros((box_voxels, box_voxels, box_voxels), dtype=np.float64)
    floor = np.floor(voxel_xyz).astype(int)
    frac = voxel_xyz - floor
    for corner in product((0, 1), repeat=3):
        index = floor + corner
        inside = np.all((index >= 0) & (index < box_voxels), axis=1)
        weight = weights * np.prod(np.where(np.array(corner) == 1, frac, 1.0 - frac), axis=1)
        np.add.at(volume, (index[inside, 0], index[inside, 1], index[inside, 2]), weight[inside])
    return volume

# simulate_density: blurred density map from a predicted structure, on the class-average grid
def simulate_density(
    structure_path: Path,
    box_voxels: int,
    voxel_size_angstrom: float, 
    resolution_angstrom: float
) -> np.ndarray:
    coordinates, weights, _elements = read_pdb_atoms(structure_path)
    oriented = orient_to_membrane_slab(coordinates, weights)
    voxel_xyz = oriented / voxel_size_angstrom + (box_voxels - 1) / 2.0
    volume = _splat(voxel_xyz, weights, box_voxels)

    sigma_voxels = resolution_angstrom * _FWHM_TO_SIGMA / voxel_size_angstrom
    blurred = gaussian_filter(volume, sigma=max(sigma_voxels, 0.5))
    peak = blurred.max()
    return (blurred / peak).astype(np.float32) if peak > 0 else blurred.astype(np.float32)

# azimuthal_smear: rotational average about z, to match class average
def azimuthal_smear(volume: np.ndarray, n_angles: int = 72) -> np.ndarray:
    accumulator = np.zeros_like(volume, dtype=np.float64)
    for angle in np.linspace(0.0, 360.0, n_angles, endpoint=False):
        accumulator += rotate(
            volume, angle, axes=(0, 1), reshape=False, order=1, mode='constant'
        )
    return (accumulator / n_angles).astype(volume.dtype)

# to_comparable: reduce density maps for comparability by band-limiting to common resolution and azimuthally averaging about z
def to_comparable(
    volume: np.ndarray,
    voxel_size_angstrom: float,
    resolution_angstrom: float,
    *,
    already_bandlimited: bool = False,
) -> np.ndarray:
    prepared = volume.astype(np.float64)
    if not already_bandlimited:
        sigma_voxels = resolution_angstrom * _FWHM_TO_SIGMA / voxel_size_angstrom
        prepared = gaussian_filter(prepared, sigma=max(sigma_voxels, 0.5))
    return azimuthal_smear(prepared).astype(np.float32)

# orient_to_membrane_slab: rotate mass-weighted coordinates so the slab normal (largest inertial moment) is +z
def orient_to_membrane_slab(coordinates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centred = coordinates - np.average(coordinates, axis=0, weights=weights)
    # mass-weighted inertia tensor (largest eigenvalue is the axis mass is spread widest around)
    inertia = np.zeros((3, 3))
    for axis_i in range(3):
        for axis_j in range(3):
            delta = 1.0 if axis_i == axis_j else 0.0
            inertia[axis_i, axis_j] = np.sum(weights * (delta * (centred ** 2).sum(axis=1) - centred[:, axis_i] * centred[:, axis_j]))
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    ordered = eigenvectors[:, np.argsort(eigenvalues)]  # columns: min, mid, max moment
    rotation = ordered.T  # maps min->x, mid->y, max(normal)->z
    if np.linalg.det(rotation) < 0:
        rotation[0] *= -1.0  # keep a proper rotation
    return centred @ rotation.T
