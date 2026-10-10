'''
STAMP: selection of picks from a saved candidate table at a target false-discovery rate
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from scipy.spatial import cKDTree

# Import internal STAMP objects
from stamp.picking.candidates import CandidateTable
from stamp.utils.log import log

# _SOURCE_CODES: candidate table source column codes
_SOURCE_CODES = {'dog': 0, 'lattice': 1}

# SelectConfig: parameters applied to a saved candidate table (changing doesn't re-run detection)
@dataclass(frozen=True)
class SelectConfig:
    target_fdr: float = 0.05
    min_zscore: float = 3.0
    shell_angstrom: tuple[float, float] | None = (0.0, 250.0)    # midplane distance range; None disables
    include_inside_membrane: bool = True
    min_r_mid: float = 0.3                                        # provisional
    max_line: float = 0.3                                         # provisional; ring definition changed
    exclude_contaminated: bool = True
    min_particle_distance_angstrom: float = 40.0
    sources: tuple[str, ...] = ('dog', 'lattice')

# Selection: kept signal row indices with the threshold and FDR curve that produced them
@dataclass(frozen=True)
class Selection:
    indices: np.ndarray
    zscore_threshold: float
    fdr_curve: np.ndarray            # (n, 4): threshold, n_signal, n_null_scaled, q_value

# select_candidates: filter, de-duplicate and threshold a candidate table at the target FDR
def select_candidates(table: CandidateTable, config: SelectConfig) -> Selection:
    voxel = table.meta['voxel_size_angstrom']
    elongation = table.meta['elongation']
    signal_kept = _filter_and_suppress(table.columns, config, voxel, elongation)
    null_kept = _filter_and_suppress(table.null_columns, config, voxel, elongation)
    signal_z = table.columns['zscore'][signal_kept]
    null_z = table.null_columns['zscore'][null_kept]

    # fdr curve: thresholds on a 0.05 grid, null counts scaled to the signal volume
    if len(signal_z) == 0:
        log.warning('No candidates survive the filters; selection is empty')
        return Selection(np.empty(0, dtype=np.int64), np.inf, np.empty((0, 4)))
    thresholds = np.arange(config.min_zscore, signal_z.max() + 0.05, 0.05)
    n_signal = (signal_z[None, :] >= thresholds[:, None]).sum(axis=1)
    n_null = table.null_scale * (null_z[None, :] >= thresholds[:, None]).sum(axis=1)
    fdr = np.minimum(1.0, n_null / np.maximum(n_signal, 1))
    # q-value: lowest false-discovery rate reached at or below a candidate's score
    q_value = np.minimum.accumulate(fdr)
    curve = np.column_stack([thresholds, n_signal, n_null, q_value])
    passing = np.flatnonzero(fdr <= config.target_fdr)
    if len(passing) == 0:
        log.warning(f'No threshold reaches the target FDR {config.target_fdr}; best FDR reached is {fdr.min():.3f}')
        return Selection(np.empty(0, dtype=np.int64), np.inf, curve)
    threshold = float(thresholds[passing[0]])
    indices = np.sort(signal_kept[signal_z >= threshold])
    return Selection(indices, threshold, curve)

# _filter_and_suppress: indices of rows that pass the feature filters and survive anisotropic NMS
def _filter_and_suppress(columns: dict[str, np.ndarray], config: SelectConfig, voxel: float, elongation: float) -> np.ndarray:
    keep = columns['zscore'] >= config.min_zscore
    midplane = columns['midplane_distance_angstrom']
    if config.shell_angstrom is not None and not np.isnan(midplane).all():
        low, high = config.shell_angstrom
        keep &= (midplane >= low) & (midplane <= high)
    if not config.include_inside_membrane:
        keep &= columns['inside_membrane'] == 0
    keep &= columns['r_mid'] >= config.min_r_mid
    # nan ring value means no valid ring, which passes
    keep &= ~(columns['line'] > config.max_line)
    if config.exclude_contaminated:
        keep &= columns['contaminated'] == 0
    keep &= np.isin(columns['source'], [_SOURCE_CODES[name] for name in config.sources])
    rows = np.flatnonzero(keep)
    if len(rows) == 0:
        return rows
    positions = np.column_stack([columns['z'][rows], columns['y'][rows], columns['x'][rows]])
    radii = np.maximum(config.min_particle_distance_angstrom / voxel, np.sqrt(3.0) * columns['sigma_voxels'][rows])
    return rows[_nms(positions, columns['zscore'][rows], radii, elongation)]

# _nms: greedy non-maximum suppression, highest score first, z compressed by the missing-wedge elongation; returns kept indices into the inputs
def _nms(positions_zyx: np.ndarray, scores: np.ndarray, radii_voxels: np.ndarray, elongation: float) -> np.ndarray:
    if len(positions_zyx) == 0:
        return np.empty(0, dtype=np.int64)
    metric = positions_zyx / np.array([elongation, 1.0, 1.0])
    tree = cKDTree(metric)
    # stable sort on -score: ties fall back to row order
    order = np.argsort(-scores, kind='stable')
    suppressed = np.zeros(len(metric), dtype=bool)
    kept: list[int] = []
    for index in order:
        if suppressed[index]:
            continue
        kept.append(int(index))
        suppressed[tree.query_ball_point(metric[index], r=radii_voxels[index])] = True
    return np.array(sorted(kept), dtype=np.int64)
