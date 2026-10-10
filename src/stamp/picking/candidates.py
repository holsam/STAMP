'''
STAMP: multiscale candidate detection (difference-of-Gaussians maxima), description in membrane coordinates and a phase-randomised noise null
'''

# Import external dependencies
import json, numpy as np, scipy.fft, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from scipy.ndimage import binary_dilation, gaussian_filter, gaussian_filter1d, generate_binary_structure, map_coordinates, maximum_filter
from scipy.spatial import cKDTree

# Import internal STAMP objects
from stamp.picking.blocks import Block, active_blocks, bin_mean, iter_blocks
from stamp.picking.geometry import extract_surface, membrane_half_thickness
from stamp.utils.checkpoint import atomic_save_npz
from stamp.utils.errors import StampValidationError
from stamp.utils.io import open_mrc, read_mrc_block, read_mrc_mask, read_mrc_shape
from stamp.utils.log import log

# PICKER_VERSION: current version of the native STAMP picker
PICKER_VERSION = 2

# _PROVISIONAL_MARGIN: a block keeps maxima down to this many MADs below the prune threshold, judged on its own statistics, so features are only computed for plausible rows
_PROVISIONAL_MARGIN = 1.0

# _MIN_BLOCK_MAXIMA: blocks with fewer maxima than this have no usable local statistics and keep every maximum
_MIN_BLOCK_MAXIMA = 50

# _RING_POINTS: samples on each of the inner ring and the outer annulus of the ring test (even, so every point has an opposite)
_RING_POINTS = 24

# _COLUMN_DTYPES: column name and dtype of every candidate table column, in storage order
_COLUMN_DTYPES: dict[str, type] = {
    'z': np.float32, 'y': np.float32, 'x': np.float32,
    'scale': np.int16, 'sigma_voxels': np.float32, 'radius_angstrom': np.float32,
    'response': np.float32, 'zscore': np.float32,
    'ev0': np.float32, 'ev1': np.float32, 'ev2': np.float32,
    'r_small': np.float32, 'r_mid': np.float32, 'long_axis_z': np.float32, 'line': np.float32,
    'boundary_distance_angstrom': np.float32, 'half_thickness_angstrom': np.float32,
    'midplane_distance_angstrom': np.float32, 'inside_membrane': np.int8,
    'normal_z': np.float32, 'normal_y': np.float32, 'normal_x': np.float32,
    'polarity_known': np.int8, 'side': np.int8, 'beam_angle_degrees': np.float32,
    'crowding': np.int16, 'contaminated': np.int8, 'vesicle_label': np.int32, 'source': np.int8,
}

# DetectConfig: parameters that change detection output (distances in Å)
@dataclass(frozen=True)
class DetectConfig:
    voxel_size_angstrom: float
    density_sign: int = -1
    radius_range_angstrom: tuple[float, float] = (15.0, 60.0)
    scale_ratio: float = 1.6
    tilt_range_degrees: float = 51.0
    membrane_guided: bool = True
    reach_angstrom: float = 300.0             # candidates further than this from the membrane are dropped when membrane_guided
    block_size: int = 128
    surface_bin: int = 2
    max_thickness_angstrom: float = 120.0
    ring_radius_factor: float = 2.5
    contamination_n_mad: float = 5.0          # flags artefacts on all three example tomograms (2_engine/contamination_results.txt)
    contamination_dilate_angstrom: float = 100.0
    null_block_fraction: float = 0.25
    crowding_radius_angstrom: float = 150.0
    crowding_min_zscore: float = 4.0
    prune_zscore: float = 2.0
    seed: int = 0

    # to_voxels: convert Angstrom to voxels
    def to_voxels(self, angstrom: float) -> float:
        return angstrom / self.voxel_size_angstrom

    # elongation: missing-wedge z elongation from tilt_range_degrees
    @property
    def elongation(self) -> float:
        alpha = np.radians(self.tilt_range_degrees)
        cross = np.sin(alpha) * np.cos(alpha)
        return float(np.sqrt((alpha + cross) / (alpha - cross)))

    # sigmas_voxels: detection scales, sigma = radius / (sqrt(3) * voxel) over radius_range_angstrom in steps of scale_ratio
    @property
    def sigmas_voxels(self) -> tuple[float, ...]:
        low, high = self.radius_range_angstrom
        radii = [low]
        while radii[-1] * self.scale_ratio <= high * 1.0001:
            radii.append(radii[-1] * self.scale_ratio)
        return tuple(radius / (np.sqrt(3.0) * self.voxel_size_angstrom) for radius in radii)

# CandidateTable: one tomogram's candidates and null candidates as named columns, plus per-scale statistics and metadata
@dataclass(frozen=True)
class CandidateTable:
    columns: dict[str, np.ndarray]
    null_columns: dict[str, np.ndarray]
    null_scale: float                         # multiply null counts by this to compare with signal counts
    scale_stats: np.ndarray                   # (n_scales, 3): sigma_voxels, median, mad of the response over all signal maxima
    meta: dict                                # picker_version, tomogram_id, shape_zyx, voxel_size_angstrom, elongation, detect_config

    # save: write the table atomically to an .npz (meta as a JSON string)
    def save(self, path: Path) -> None:
        arrays = {f'signal__{name}': values for name, values in self.columns.items()}
        arrays.update({f'null__{name}': values for name, values in self.null_columns.items()})
        atomic_save_npz(
            path, scale_stats=self.scale_stats, null_scale=np.float64(self.null_scale),
            meta=np.array(json.dumps(self.meta)), **arrays,
        )

    # load: read a table written by save
    @classmethod
    def load(cls, path: Path) -> 'CandidateTable':
        with np.load(path, allow_pickle=False) as data:
            columns = {key[len('signal__'):]: data[key] for key in data.files if key.startswith('signal__')}
            null_columns = {key[len('null__'):]: data[key] for key in data.files if key.startswith('null__')}
            return cls(columns, null_columns, float(data['null_scale']), data['scale_stats'], json.loads(str(data['meta'])))

    # __len__: number of signal candidates
    def __len__(self) -> int:
        return len(self.columns['z'])

# detect_candidates: geometry, signal and null detection, and description for one tomogram
def detect_candidates(
    segmentation_path: Path,
    tomogram_path: Path,
    tomogram_id: str,
    config: DetectConfig,
    *,
    n_threads: int = 1,
    vesicle_labels_mrc: Path | None = None,
) -> CandidateTable:
    shape = read_mrc_shape(tomogram_path)
    mask = read_mrc_mask(segmentation_path)
    if mask.shape != shape:
        raise StampValidationError(f'Segmentation {str(segmentation_path)!r} has shape {mask.shape}, tomogram {str(tomogram_path)!r} has shape {shape}')

    # geometry: the mesh is built whenever the mask has voxels so unguided runs still get membrane features
    geometry = _Geometry.build(mask, config) if (config.membrane_guided or mask.any()) else None

    scales = _make_scales(config)
    halo = _halo_voxels(scales, config)
    max_factor = 2 ** max(scale.level for scale in scales)
    blocks = iter_blocks(shape, config.block_size, halo, align=max_factor)
    if config.membrane_guided:
        blocks = active_blocks(mask, blocks, config.to_voxels(config.reach_angstrom) + halo)
    log.debug(f'{tomogram_id}: {len(scales)} scale(s), halo {halo} voxels, {len(blocks)} active block(s)')

    null_count = min(len(blocks), max(1, round(config.null_block_fraction * len(blocks)))) if blocks else 0
    null_blocks = [blocks[i] for i in sorted(np.random.default_rng(config.seed).choice(len(blocks), null_count, replace=False))] if blocks else []

    started = time.time()
    with ThreadPoolExecutor(n_threads) as pool:
        signal_results = list(pool.map(lambda block: _detect_block(block, tomogram_path, mask, config, scales, null=False), blocks))
        null_results = list(pool.map(lambda block: _detect_block(block, tomogram_path, mask, config, scales, null=True), null_blocks))
    log.debug(f'{tomogram_id}: signal and null passes took {time.time() - started:.1f} s')
    signal_rows = _stack_rows([rows for rows, _responses in signal_results])
    null_rows = _stack_rows([rows for rows, _responses in null_results])
    null_scale = _core_voxels(blocks) / max(_core_voxels(null_blocks), 1)

    # statistics: per-scale median and MAD over every signal maximum, applied to signal and null alike
    scale_stats = np.zeros((len(scales), 3))
    for index, scale in enumerate(scales):
        responses = np.concatenate([result[1][index] for result in signal_results]) if signal_results else np.empty(0)
        median = float(np.median(responses)) if len(responses) else 0.0
        mad = 1.4826 * float(np.median(np.abs(responses - median))) if len(responses) else 1.0
        scale_stats[index] = (scale.sigma, median, mad if mad > 0 else 1.0)
    signal_rows = _prune(signal_rows, scale_stats, config.prune_zscore)
    null_rows = _prune(null_rows, scale_stats, config.prune_zscore)

    contamination = contamination_mask(tomogram_path, config)
    columns = _describe(signal_rows, scale_stats, mask, geometry, contamination, config, shape, n_threads, vesicle_labels_mrc)
    null_columns = _describe(null_rows, scale_stats, mask, geometry, contamination, config, shape, n_threads, None)
    meta = {
        'picker_version': PICKER_VERSION, 'tomogram_id': tomogram_id, 'shape_zyx': list(shape),
        'voxel_size_angstrom': config.voxel_size_angstrom, 'elongation': config.elongation,
        'detect_config': asdict(config),
    }
    log.debug(f'{tomogram_id}: {len(columns["z"])} signal and {len(null_columns["z"])} null candidate(s)')
    return CandidateTable(columns, null_columns, float(null_scale), scale_stats, meta)

# contamination_mask: boolean mask at bin 4 of voxels whose coarse density is an extreme outlier, dilated
def contamination_mask(tomogram_path: Path, config: DetectConfig) -> np.ndarray:
    factor = 4
    shape = tuple(n // factor for n in read_mrc_shape(tomogram_path))
    binned = np.empty(shape, dtype=np.float32)
    slab = 4 * factor
    for start in range(0, shape[0] * factor, slab):
        stop = min(start + slab, shape[0] * factor)
        # a fresh map per slab keeps mapped file pages from accumulating in the resident set
        part = read_mrc_block(tomogram_path, (start, 0, 0), (stop, shape[1] * factor, shape[2] * factor))
        binned[start // factor:stop // factor] = bin_mean(part, factor)
    smooth = config.density_sign * gaussian_filter(binned, sigma=2, mode='nearest')
    median = np.median(smooth)
    mad = 1.4826 * np.median(np.abs(smooth - median))
    outliers = smooth > median + config.contamination_n_mad * (mad if mad > 0 else 1.0)
    if not outliers.any():
        return outliers
    iterations = int(np.ceil(config.to_voxels(config.contamination_dilate_angstrom) / factor))
    # scipy treats iterations=0 as 'repeat until stable', which would flood the volume
    if iterations < 1:
        return outliers
    return binary_dilation(outliers, structure=generate_binary_structure(3, 1), iterations=iterations)

# _Geometry: membrane mesh in full-resolution zyx voxels with a k-d tree over its vertices
@dataclass(frozen=True)
class _Geometry:
    vertices: np.ndarray
    normals: np.ndarray
    tree: cKDTree

    # build: extract the (binned) surface of a mask
    @classmethod
    def build(cls, mask: np.ndarray, config: DetectConfig) -> '_Geometry':
        vertices, normals = extract_surface(mask, bin_factor=config.surface_bin)
        return cls(vertices, normals, cKDTree(vertices))

# _Scale: one detection scale and the binning level it runs at (sigmas in voxels at the level unless noted)
@dataclass(frozen=True)
class _Scale:
    index: int
    sigma: float                 # full-resolution voxels
    level: int
    sigma_level: float
    next_sigma_level: float      # larger Gaussian of the DoG
    radius_level: int            # maximum-filter radius
    ring_radius_level: float     # inner ring radius

# _make_scales: scales with the pyramid level at which each sigma is 1.5 to 3 voxels
def _make_scales(config: DetectConfig) -> list[_Scale]:
    scales = []
    for index, sigma in enumerate(config.sigmas_voxels):
        level = max(0, int(np.floor(np.log2(sigma / 1.5))))
        factor = 2 ** level
        sigma_level = sigma / factor
        scales.append(_Scale(
            index, sigma, level, sigma_level, sigma_level * config.scale_ratio,
            max(1, round(sigma_level)), (config.ring_radius_factor * sigma + 2) / factor,
        ))
    return scales

# _halo_voxels: halo that makes block results match a whole-volume run away from the volume edges, a multiple of the coarsest binning
def _halo_voxels(scales: list[_Scale], config: DetectConfig) -> int:
    halo = 0
    for scale in scales:
        factor = 2 ** scale.level
        reach = np.ceil(3 * scale.sigma * config.scale_ratio) + scale.radius_level * factor + np.ceil(2 * scale.ring_radius_level * factor)
        halo = max(halo, int(reach) + 2)
    factor = 2 ** max(scale.level for scale in scales)
    return -(-halo // factor) * factor

# _core_voxels: total core volume of a list of blocks
def _core_voxels(blocks: list[Block]) -> int:
    return sum(int(np.prod(np.subtract(block.core_hi, block.core_lo))) for block in blocks)

# _detect_block: maxima of every DoG scale inside one block's core, with Hessian and ring features, plus the responses of all core maxima per scale
def _detect_block(
    block: Block, tomogram_path: Path, mask: np.ndarray, config: DetectConfig, scales: list[_Scale], *, null: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:
    v = read_mrc_block(tomogram_path, block.read_lo, block.read_hi)
    if config.density_sign < 0:
        np.negative(v, out=v)
    if null:
        # random phases keep the amplitude spectrum (noise, CTF, missing wedge) and destroy structure; scipy.fft transforms float32 natively, numpy's upcasts to float64
        rng = np.random.default_rng([config.seed, block.index])
        shape = v.shape
        v -= v.mean()
        spectrum = scipy.fft.rfftn(v, axes=(0, 1, 2))
        del v
        amplitude = np.abs(spectrum)
        del spectrum
        phase = rng.random(amplitude.shape, dtype=np.float32)
        phase *= np.float32(2 * np.pi)
        spectrum = np.empty(amplitude.shape, dtype=np.complex64)
        spectrum.real = np.cos(phase)
        spectrum.imag = np.sin(phase)
        del phase
        spectrum *= amplitude
        del amplitude
        v = scipy.fft.irfftn(spectrum, s=shape, axes=(0, 1, 2)).astype(np.float32, copy=False)
        del spectrum
    read_lo = np.array(block.read_lo)
    rows: list[np.ndarray] = []
    responses = [np.empty(0, dtype=np.float32) for _ in scales]
    for level in sorted({scale.level for scale in scales}):
        factor = 2 ** level
        level_volume = bin_mean(v, factor)
        # core in level coordinates: p * factor + (factor - 1) / 2 + read_lo lies in [core_lo, core_hi)
        shift = (factor - 1) / 2
        core_lo = np.ceil((np.array(block.core_lo) - read_lo - shift) / factor).astype(int)
        core_hi = np.ceil((np.array(block.core_hi) - read_lo - shift) / factor).astype(int)
        level_scales = [scale for scale in scales if scale.level == level]
        # smoothed volumes are only kept over the core plus what the filter, Hessian and ring samples reach
        margin = max(max(scale.radius_level, int(np.ceil(2 * scale.ring_radius_level)) + 2, 2) for scale in level_scales)
        region = tuple(slice(max(0, lo - margin), min(n, hi + margin)) for lo, hi, n in zip(core_lo, core_hi, level_volume.shape))
        origin = np.array([part.start for part in region])
        local_lo, local_hi = core_lo - origin, core_hi - origin
        smoothed: dict[float, np.ndarray] = {}
        for scale in level_scales:
            # only the current and next scale's Gaussians are kept to bound memory
            smoothed = {key: value for key, value in smoothed.items() if key in (_key(scale.sigma_level), _key(scale.next_sigma_level))}
            for sigma in (scale.sigma_level, scale.next_sigma_level):
                if _key(sigma) not in smoothed:
                    smoothed[_key(sigma)] = _smooth(level_volume, sigma, region)
            fine, coarse = smoothed[_key(scale.sigma_level)], smoothed[_key(scale.next_sigma_level)]
            # the DoG and its maximum filter only need the core plus the filter radius
            extended = tuple(slice(max(0, lo - scale.radius_level), hi + scale.radius_level) for lo, hi in zip(local_lo, local_hi))
            dog = fine[extended] - coarse[extended]
            inner = tuple(slice(lo - e.start, hi - e.start) for e, lo, hi in zip(extended, local_lo, local_hi))
            dog_core = dog[inner]
            core_peaks = (dog_core == maximum_filter(dog, size=2 * scale.radius_level + 1, mode='nearest')[inner]) & (dog_core > 0)
            all_responses = dog_core[core_peaks]
            responses[scale.index] = all_responses
            if len(all_responses) >= _MIN_BLOCK_MAXIMA:
                median = np.median(all_responses)
                threshold = median + max(config.prune_zscore - _PROVISIONAL_MARGIN, 0.0) * 1.4826 * np.median(np.abs(all_responses - median))
                core_peaks = core_peaks & (dog_core > threshold)
            peak_indices = np.argwhere(core_peaks)
            if len(peak_indices) == 0:
                continue
            response = dog_core[tuple(peak_indices.T)]
            indices = peak_indices + local_lo
            eigenvalues, long_axis_z = _hessian(fine, indices)
            line = _ring_line(fine, indices, scale, factor, origin * factor + read_lo, mask, config)
            positions = (indices + origin) * factor + shift + read_lo
            rows.append(np.column_stack([
                positions, np.full(len(indices), scale.index), response, eigenvalues, long_axis_z, line,
            ]).astype(np.float32))
        del smoothed, level_volume
    return (np.concatenate(rows) if rows else np.empty((0, 10), dtype=np.float32)), responses

# _key: dictionary key for a Gaussian width, tolerant of floating-point noise
def _key(sigma: float) -> float:
    return round(float(sigma), 6)

# _smooth: Gaussian filter evaluated only over region (slices into volume)
def _smooth(volume: np.ndarray, sigma: float, region: tuple[slice, ...]) -> np.ndarray:
    support = int(3.0 * sigma + 0.5)
    # per-axis extent of the current array in volume coordinates
    current = volume
    lows = [0, 0, 0]
    for axis in range(3):
        # crop to what this and the remaining passes need: region plus support on axes not yet filtered
        crop = []
        for other in range(3):
            pad = support if other >= axis else 0
            lo = max(0, region[other].start - pad)
            hi = min(volume.shape[other], region[other].stop + pad)
            crop.append(slice(lo - lows[other], hi - lows[other]))
            lows[other] = lo
        current = gaussian_filter1d(current[tuple(crop)], sigma, axis=axis, mode='nearest', truncate=3.0)
    return current[tuple(slice(part.start - low, part.stop - low) for part, low in zip(region, lows))]

# _hessian: ascending eigenvalues of -H of the smoothed volume at integer peak indices, and |z| of the smallest-curvature eigenvector
def _hessian(smoothed: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = np.clip(indices, 1, np.array(smoothed.shape) - 2)
    identity = np.eye(3, dtype=int)

    # sample: smoothed values at the peaks shifted by an offset
    def sample(offset: np.ndarray) -> np.ndarray:
        return smoothed[tuple((centre + offset).T)]

    hessian = np.empty((len(indices), 3, 3), dtype=np.float32)
    middle = sample(np.zeros(3, dtype=int))
    for a in range(3):
        hessian[:, a, a] = sample(identity[a]) - 2 * middle + sample(-identity[a])
        for b in range(a + 1, 3):
            hessian[:, a, b] = hessian[:, b, a] = (
                sample(identity[a] + identity[b]) - sample(identity[a] - identity[b])
                - sample(identity[b] - identity[a]) + sample(-identity[a] - identity[b])
            ) / 4
    eigenvalues, eigenvectors = np.linalg.eigh(-hessian)
    return eigenvalues, np.abs(eigenvectors[:, 0, 0])

# _ring_line: ring test at each peak's z, the strongest opposite pair on an inner ring relative to a local annulus background, as a fraction of the centre contrast
def _ring_line(
    smoothed: np.ndarray, indices: np.ndarray, scale: _Scale, factor: int, offset_full: np.ndarray, mask: np.ndarray, config: DetectConfig,
) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, _RING_POINTS, endpoint=False)
    n = len(indices)
    samples = np.empty((n, 2, _RING_POINTS))
    valid = np.empty((n, 2, _RING_POINTS), dtype=bool)
    for ring, radius in enumerate((scale.ring_radius_level, 2 * scale.ring_radius_level)):
        z = np.repeat(indices[:, 0:1], _RING_POINTS, axis=1).astype(float)
        y = indices[:, 1:2] + radius * np.sin(angles)[None, :]
        x = indices[:, 2:3] + radius * np.cos(angles)[None, :]
        samples[:, ring] = map_coordinates(smoothed, np.stack([z, y, x]).reshape(3, -1), order=1, mode='nearest').reshape(n, _RING_POINTS)
        # ring samples inside the segmentation are ignored (R9)
        full = np.rint(np.stack([z, y, x], axis=-1) * factor + (factor - 1) / 2 + offset_full).astype(int)
        full = np.clip(full, 0, np.array(mask.shape) - 1)
        valid[:, ring] = ~mask[full[..., 0], full[..., 1], full[..., 2]]
    inner, outer = samples[:, 0], samples[:, 1]
    outer_valid = valid[:, 1]
    background = np.where(outer_valid, outer, np.nan)
    with np.errstate(all='ignore'):
        background = np.nanmedian(np.where(outer_valid.any(axis=1)[:, None], background, 0.0), axis=1)
    contrast = smoothed[tuple(indices.T)] - background
    half = _RING_POINTS // 2
    pair_valid = valid[:, 0, :half] & valid[:, 0, half:]
    pair_mean = 0.5 * ((inner[:, :half] - background[:, None]) + (inner[:, half:] - background[:, None]))
    pair_mean = np.where(pair_valid, pair_mean, -np.inf)
    with np.errstate(all='ignore'):
        line = pair_mean.max(axis=1) / contrast
    usable = (pair_valid.sum(axis=1) >= 4) & outer_valid.any(axis=1) & (contrast > 0)
    return np.where(usable, line, np.nan)

# _stack_rows: concatenate per-block raw rows
def _stack_rows(parts: list[np.ndarray]) -> np.ndarray:
    parts = [part for part in parts if len(part)]
    return np.concatenate(parts) if parts else np.empty((0, 10), dtype=np.float32)

# _prune: keep raw rows whose z-score under the signal statistics exceeds the prune threshold
def _prune(rows: np.ndarray, scale_stats: np.ndarray, prune_zscore: float) -> np.ndarray:
    return rows[_zscore(rows, scale_stats) > prune_zscore]

# _zscore: (response - median) / mad with the per-scale signal statistics
def _zscore(rows: np.ndarray, scale_stats: np.ndarray) -> np.ndarray:
    scale = rows[:, 3].astype(int)
    return (rows[:, 4] - scale_stats[scale, 1]) / scale_stats[scale, 2]

# _describe: per-candidate features in membrane coordinates, as named columns
def _describe(
    rows: np.ndarray,
    scale_stats: np.ndarray,
    mask: np.ndarray,
    geometry: _Geometry | None,
    contamination: np.ndarray,
    config: DetectConfig,
    shape: tuple[int, int, int],
    n_threads: int,
    vesicle_labels_mrc: Path | None,
) -> dict[str, np.ndarray]:
    voxel = config.voxel_size_angstrom
    if geometry is not None and config.membrane_guided:
        distance, nearest = geometry.tree.query(rows[:, :3], distance_upper_bound=config.to_voxels(config.reach_angstrom), workers=n_threads)
        rows = rows[np.isfinite(distance)]
        distance, nearest = distance[np.isfinite(distance)], nearest[np.isfinite(distance)]
    elif geometry is not None:
        distance, nearest = geometry.tree.query(rows[:, :3], workers=n_threads)
    n = len(rows)
    positions = rows[:, :3].astype(np.float64)
    scale = rows[:, 3].astype(int)
    sigma = scale_stats[scale, 0]
    ev0, ev1, ev2 = rows[:, 5], rows[:, 6], rows[:, 7]
    with np.errstate(all='ignore'):
        r_small, r_mid = np.where(ev2 > 0, ev0 / ev2, np.nan), np.where(ev2 > 0, ev1 / ev2, np.nan)

    nan = np.full(n, np.nan)
    boundary, half_thickness, midplane, beam, vesicle = nan.copy(), nan.copy(), nan.copy(), nan.copy(), np.zeros(n)
    normal = np.full((n, 3), np.nan)
    index = np.clip(np.rint(positions).astype(int), 0, np.array(shape) - 1)
    inside = mask[index[:, 0], index[:, 1], index[:, 2]]
    if geometry is not None and n:
        vertices, normals = geometry.vertices[nearest], geometry.normals[nearest]
        boundary = np.where(inside, -distance, distance) * voxel
        unique, inverse = np.unique(nearest, return_inverse=True)
        thickness = membrane_half_thickness(
            mask, geometry.vertices[unique], geometry.normals[unique], max_voxels=config.to_voxels(config.max_thickness_angstrom),
        )
        half_thickness = thickness[inverse] * voxel
        midplane = np.maximum(0.0, half_thickness + boundary)
        # outside: normal points from the membrane towards the candidate; inside: outward face direction
        towards = np.einsum('ij,ij->i', normals, positions - vertices) < 0
        probe = np.clip(np.rint(vertices - 1.5 * normals).astype(int), 0, np.array(shape) - 1)
        probe_inside = mask[probe[:, 0], probe[:, 1], probe[:, 2]]
        flip = np.where(inside, ~probe_inside, towards)
        normal = np.where(flip[:, None], -normals, normals)
        # same quantity as geometry.beam_angle_deviation_degrees, vectorised; degenerate zero normals give nan
        beam = np.degrees(np.arcsin(np.clip(np.abs(normal[:, 0]), 0.0, 1.0)))
        if vesicle_labels_mrc is not None:
            vesicle = _vesicle_labels_at(vesicle_labels_mrc, geometry.vertices[nearest], shape)

    # crowding in the anisotropic metric, counting only strong neighbours
    zscore = _zscore(rows, scale_stats)
    crowding = np.zeros(n)
    strong = zscore >= config.crowding_min_zscore
    if strong.any():
        metric = positions / np.array([config.elongation, 1.0, 1.0])
        counts = cKDTree(metric[strong]).query_ball_point(metric, r=config.to_voxels(config.crowding_radius_angstrom), return_length=True)
        crowding = counts - strong

    coarse = np.clip(np.rint(positions / 4).astype(int), 0, np.array(contamination.shape) - 1)
    contaminated = contamination[coarse[:, 0], coarse[:, 1], coarse[:, 2]] if n else np.zeros(0)

    values = {
        'z': positions[:, 0], 'y': positions[:, 1], 'x': positions[:, 2],
        'scale': scale, 'sigma_voxels': sigma, 'radius_angstrom': np.sqrt(3.0) * sigma * voxel,
        'response': rows[:, 4], 'zscore': zscore,
        'ev0': ev0, 'ev1': ev1, 'ev2': ev2, 'r_small': r_small, 'r_mid': r_mid,
        'long_axis_z': rows[:, 8], 'line': rows[:, 9],
        'boundary_distance_angstrom': boundary, 'half_thickness_angstrom': half_thickness,
        'midplane_distance_angstrom': midplane, 'inside_membrane': inside,
        'normal_z': normal[:, 0], 'normal_y': normal[:, 1], 'normal_x': normal[:, 2],
        'polarity_known': np.zeros(n), 'side': np.full(n, -1), 'beam_angle_degrees': beam,
        'crowding': crowding, 'contaminated': contaminated, 'vesicle_label': vesicle, 'source': np.zeros(n),
    }
    return {name: np.asarray(values[name]).astype(dtype) for name, dtype in _COLUMN_DTYPES.items()}

# _vesicle_labels_at: integer EValuator label at each point (nearest voxel), read through a memory map so the label volume is never loaded
def _vesicle_labels_at(labels_mrc: Path, points_zyx: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    with open_mrc(labels_mrc, mmap=True) as mrc:
        if tuple(mrc.data.shape) != shape:
            raise StampValidationError(f'Vesicle labels MRC {str(labels_mrc)!r} has shape {tuple(mrc.data.shape)}, expected {shape} to match the segmentation')
        index = np.clip(np.rint(points_zyx).astype(np.int64), 0, np.array(shape) - 1)
        return np.asarray(mrc.data[index[:, 0], index[:, 1], index[:, 2]]).astype(np.int64)
