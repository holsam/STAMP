'''
STAMP: 2D mocked functions for identify
'''

# Import external dependencies
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate

# --- PDB structure -> illustrative 2D density projection ---
# read_atom_coordinates: read (N, 3) Cartesian coordinates from ATOM/HETATM records, fixed-column PDB format with a whitespace-split fallback
def read_atom_coordinates(pdb_path: Path) -> np.ndarray:
    coordinates = []
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            fields = line.split()
            x, y, z = (float(v) for v in fields[6:9])
        coordinates.append((x, y, z))
    if not coordinates:
        raise ValueError(f'no ATOM/HETATM records in {pdb_path}')
    return np.array(coordinates)

# project_density: project atoms down the z axis onto a 2D grid and blur each point to a plausible blob
def project_density(
    coordinates: np.ndarray,
    box: int,
    pixel_size: float,
    in_plane_deg: float = 0.0,
    atom_sigma_px: float = 1.1,
) -> np.ndarray:
    centred = coordinates - coordinates.mean(axis=0)
    xy = centred[:, :2] / pixel_size
    if in_plane_deg:
        theta = np.deg2rad(in_plane_deg)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xy = xy @ rotation.T
    half = box / 2.0
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=box, range=[[-half, half], [-half, half]])
    density = gaussian_filter(hist, sigma=atom_sigma_px)
    peak = density.max()
    return density / peak if peak > 0 else density


# --- deterministic distinctive 'structures' for species with no PDB ---
# _UNKNOWN_TEMPLATES: each has its own shape and spatial-frequency content, so they cluster cleanly and score low against the smooth real candidates
_UNKNOWN_TEMPLATES = ('trimer', 'bar', 'ring', 'tetramer', 'sshape')

# synthetic_truth: a deterministic, deliberately distinctive 'structure'
def synthetic_truth(box: int, index: int) -> np.ndarray:
    yy, xx = np.mgrid[0:box, 0:box]
    half = (box - 1) / 2.0
    y, x = yy - half, xx - half
    radius = np.hypot(x, y)
    angle = np.arctan2(y, x)
    kind = _UNKNOWN_TEMPLATES[index % len(_UNKNOWN_TEMPLATES)]

    if kind == 'bar':
        image = np.exp(-(x ** 2 / (2 * (0.30 * box) ** 2) + y ** 2 / (2 * (0.06 * box) ** 2)))
    elif kind == 'ring':
        image = np.exp(-((radius - 0.24 * box) ** 2) / (2 * (0.05 * box) ** 2))
    elif kind == 'sshape':
        image = np.zeros((box, box))
        for sign in (-1, 1):
            cx, cy = sign * 0.12 * box, sign * 0.12 * box
            image += np.exp(-((np.hypot(x - cx, y - cy) - 0.16 * box) ** 2) / (2 * (0.05 * box) ** 2)) * (np.abs(angle) < 2.4)
    else:  # trimer / tetramer: n lobes on a circle
        n = 3 if kind == 'trimer' else 4
        image = np.zeros((box, box))
        for k in range(n):
            phi = 2 * np.pi * k / n
            cx, cy = 0.20 * box * np.cos(phi), 0.20 * box * np.sin(phi)
            image += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (0.09 * box) ** 2))

    image = gaussian_filter(image, sigma=1.0)
    peak = image.max()
    return image / peak if peak > 0 else image
