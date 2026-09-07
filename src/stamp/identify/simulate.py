'''
STAMP: predicted structure to density map
'''

# Import external dependencies
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate

# _ELEMENT_WEIGHT: rough scattering weight per element (atomic number is close enough at this resolution)
_ELEMENT_WEIGHT = {'H': 1.0, 'C': 6.0, 'N': 7.0, 'O': 8.0, 'P': 15.0, 'S': 16.0}
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
        element = (line[76:78].strip() or line[12:16].strip()[:1] or 'C').upper()
        elements.append(element)
        weights.append(_ELEMENT_WEIGHT.get(element, 6.0))
    if not coordinates:
        raise ValueError(f'{structure_path!r} has no ATOM/HETATM records')
    return np.array(coordinates), np.array(weights), elements

# _splat: accumulate per-atom weight into the volume (nearest voxel)
def _splat(voxel_xyz: np.ndarray, weights: np.ndarray, box_voxels: int) -> np.ndarray:
    nearest = np.rint(voxel_xyz).astype(int)
    inside = np.all((nearest >= 0) & (nearest < box_voxels), axis=1)
    nearest, kept_weights = nearest[inside], weights[inside]
    volume = np.zeros((box_voxels, box_voxels, box_voxels), dtype=np.float64)
    np.add.at(volume, (nearest[:, 0], nearest[:, 1], nearest[:, 2]), kept_weights)
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
