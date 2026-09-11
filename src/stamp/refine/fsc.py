'''
STAMP: FSC between two independent half-maps
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# FSCResult: both curves plus the resolution read at the threshold
@dataclass
class FSCResult:
    frequencies_per_angstrom: np.ndarray
    fsc_unmasked: np.ndarray
    fsc_masked: np.ndarray
    resolution_angstrom: float

# soft_sphere_mask: soft-edged spherical mask with a cosine falloff
def soft_sphere_mask(
    shape: tuple[int, int, int],
    radius_fraction: float = 0.8,
    falloff_voxels: int = 5
) -> np.ndarray:
    centre = (np.array(shape) - 1) / 2.0
    grid = np.indices(shape).astype(float) - centre[:, None, None, None]
    radius = np.sqrt((grid ** 2).sum(axis=0))
    inner = radius_fraction * min(shape) / 2.0 - falloff_voxels
    mask = np.ones(shape)
    taper = (radius - inner) / max(falloff_voxels, 1)
    band = (taper > 0) & (taper < 1)
    mask[band] = 0.5 * (1 + np.cos(np.pi * taper[band]))
    mask[taper >= 1] = 0.0
    return mask

# _shell_map: integer shell index per Fourier voxel, DC at index 0 to match np.fft.fftn
def _shell_map(shape: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    centre = np.array(shape) // 2
    grid = np.indices(shape) - centre[:, None, None, None]
    radius = np.sqrt((grid ** 2).sum(axis=0))
    return np.fft.ifftshift(radius).astype(int), int(centre.min())

# _shell_fsc: shell-wise Fourier ring correlation between two maps
def _shell_fsc(map_a: np.ndarray, map_b: np.ndarray, shell: np.ndarray, n_shells: int) -> np.ndarray:
    fft_a = np.fft.fftn(map_a)
    fft_b = np.fft.fftn(map_b)
    correlation = np.zeros(n_shells)
    for index in range(n_shells):
        selection = shell == index
        cross = np.sum(fft_a[selection] * np.conj(fft_b[selection]))
        power = np.sqrt(np.sum(np.abs(fft_a[selection]) ** 2) * np.sum(np.abs(fft_b[selection]) ** 2))
        correlation[index] = np.real(cross / power) if power > 0 else 0.0
    return correlation

# _phase_randomise: keep shell amplitudes, randomise phases at or beyond from_shell
def _phase_randomise(
    map_array: np.ndarray, shell: np.ndarray, from_shell: int, rng: np.random.Generator
) -> np.ndarray:
    transform = np.fft.fftn(map_array)
    beyond = shell >= from_shell
    phases = rng.uniform(0.0, 2.0 * np.pi, size=transform.shape)
    transform[beyond] = np.abs(transform[beyond]) * np.exp(1j * phases[beyond])
    return np.real(np.fft.ifftn(transform))

# _first_below: index of the first shell whose value drops under level, or None
def _first_below(curve: np.ndarray, level: float) -> int | None:
    hits = np.argwhere(curve < level).ravel()
    return int(hits[0]) if hits.size else None

# _resolution_at: Angstrom at the first sub-threshold shell, else the finest resolved shell
def _resolution_at(curve: np.ndarray, frequencies: np.ndarray, threshold: float) -> float:
    below = np.argwhere(curve < threshold).ravel()
    if below.size and below[0] > 0:
        return float(1.0 / frequencies[below[0]])
    return float(1.0 / frequencies[-1]) if frequencies[-1] > 0 else float('inf')

# compute_fsc: FSC (unmasked + phase-randomisation-corrected masked) with the resolution readout
def compute_fsc(
    half_map_a: np.ndarray,
    half_map_b: np.ndarray,
    voxel_size_angstrom: float,
    mask: np.ndarray | None = None,
    threshold: float = 0.143,
    *,
    seed: int = 0,
) -> FSCResult:
    if half_map_a.shape != half_map_b.shape:
        raise ValueError('half maps must have the same shape')
    if mask is None:
        mask = soft_sphere_mask(half_map_a.shape)

    shell, n_shells = _shell_map(half_map_a.shape)
    unmasked = _shell_fsc(half_map_a, half_map_b, shell, n_shells)
    masked = _shell_fsc(half_map_a * mask, half_map_b * mask, shell, n_shells)

    corrected = masked.copy()
    randomise_from = _first_below(unmasked, 0.8)
    if randomise_from is not None and randomise_from + 1 < n_shells:
        rng = np.random.default_rng(seed)
        rand_a = _phase_randomise(half_map_a, shell, randomise_from, rng) * mask
        rand_b = _phase_randomise(half_map_b, shell, randomise_from, rng) * mask
        fsc_rand = _shell_fsc(rand_a, rand_b, shell, n_shells)
        high = np.arange(n_shells) > randomise_from
        denom = np.clip(1.0 - fsc_rand, 1e-3, None)
        corrected[high] = np.clip((masked[high] - fsc_rand[high]) / denom[high], -1.0, 1.0)

    box_voxels = half_map_a.shape[-1]
    frequencies = np.arange(n_shells) / (box_voxels * voxel_size_angstrom)
    return FSCResult(
        frequencies_per_angstrom=frequencies,
        fsc_unmasked=unmasked,
        fsc_masked=corrected,
        resolution_angstrom=_resolution_at(corrected, frequencies, threshold),
    )

# write_fsc_files: fsc.txt (columns) and a minimal fsc.svg line plot
def write_fsc_files(result: FSCResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / 'fsc.txt'
    rows = ['# freq_per_A  fsc_unmasked  fsc_masked']
    for frequency, unmasked, masked in zip(result.frequencies_per_angstrom, result.fsc_unmasked, result.fsc_masked):
        rows.append(f'{frequency:.6f}  {unmasked:.4f}  {masked:.4f}')
    text_path.write_text('\n'.join(rows) + '\n')

    svg_path = output_dir / 'fsc.svg'
    width, height = 400, 240
    x = result.frequencies_per_angstrom
    x_scaled = width * (x - x.min()) / (np.ptp(x) or 1)
    points = ' '.join(
        f'{px:.1f},{height * (1 - max(min(value, 1.0), -0.2) / 1.2):.1f}'
        for px, value in zip(x_scaled, result.fsc_masked)
    )
    threshold_y = height * (1 - 0.143 / 1.2)
    svg_path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<polyline fill="none" stroke="black" points="{points}"/>'
        f'<line x1="0" y1="{threshold_y:.1f}" x2="{width}" y2="{threshold_y:.1f}" '
        f'stroke="red" stroke-dasharray="4"/>'
        f'<text x="4" y="14">FSC (masked), resolution {result.resolution_angstrom:.1f} A</text>'
        f'</svg>\n'
    )
    return text_path, svg_path
