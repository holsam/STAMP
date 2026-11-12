'''
STAMP: real spherical harmonics, sphere quadrature and masked ridge fits
'''

# Import external dependencies
import numpy as np
from scipy.special import sph_harm_y

# sphere_grid: Gauss-Legendre by uniform-azimuth grid on the unit sphere, unit vectors (n, 3) as (x, y, z) and quadrature weights summing to 4 pi
def sphere_grid(n_theta: int = 16, n_phi: int = 32) -> tuple[np.ndarray, np.ndarray]:
    nodes, gauss_weights = np.polynomial.legendre.leggauss(n_theta)
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    cos_theta, azimuth = np.meshgrid(nodes, phi, indexing='ij')
    sin_theta = np.sqrt(1 - cos_theta**2)
    xyz = np.stack([sin_theta * np.cos(azimuth), sin_theta * np.sin(azimuth), cos_theta], axis=-1).reshape(-1, 3)
    return xyz, np.repeat(gauss_weights, n_phi) * (2 * np.pi / n_phi)

# lm_pairs: (l, m) for l up to lmax, even l only when even_only
def lm_pairs(lmax: int, *, even_only: bool) -> list[tuple[int, int]]:
    return [(l, m) for l in range(0, lmax + 1, 2 if even_only else 1) for m in range(-l, l + 1)]

# real_sh_basis: orthonormal real spherical harmonics at unit vectors, shape (n_points, n_lm)
def real_sh_basis(xyz: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    theta = np.arccos(np.clip(xyz[:, 2], -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    columns = []
    for l, m in pairs:
        y = sph_harm_y(l, abs(m), theta, phi)
        columns.append(y.real if m == 0 else np.sqrt(2) * (y.real if m > 0 else y.imag) * (-1) ** abs(m))
    return np.stack(columns, axis=1)

# fit_matrix: ridge least-squares operator (n_lm, n_points) so coefficients = operator @ samples, with per-point weights (zero for missing data)
def fit_matrix(basis: np.ndarray, weights: np.ndarray, ridge: float) -> np.ndarray:
    weighted = basis * weights[:, None]
    return np.linalg.solve(basis.T @ weighted + ridge * np.eye(basis.shape[1]), weighted.T)

# degree_power: rotation invariants P_l = sum_m c_lm^2 for each l, from coefficients (..., n_lm)
def degree_power(coefficients: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    degrees = np.array([l for l, _ in pairs])
    return np.stack([(coefficients[..., degrees == l] ** 2).sum(-1) for l in np.unique(degrees)], axis=-1)
