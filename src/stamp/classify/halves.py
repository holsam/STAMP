'''
STAMP: half-set independent polish step and reference-based matching of clusters
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.polish import PolishConfig, PolishResult, run_polish
from stamp.classify.match import align_to_reference
from stamp.classify.merge import pair_thresholds
from stamp.classify.store import ParticleStore
from stamp.utils.log import log

# HalfMatch: a half-B cluster matched to a half-A cluster, with pose of the half-B reference against half-A one
@dataclass(frozen=True)
class HalfMatch:
    label_a: int
    similarity: float
    rotation: np.ndarray    # (3, 3)
    shift: np.ndarray       # (3,), working voxels

# match_halves: pair half-B references to half-A references by best-pose similarity, greedily from most similar pair, keeping only pairs that agree at least as well as references agree with themselves
def match_halves(result_a: PolishResult, result_b: PolishResult, merge_fraction: float, *, threads: int = 1) -> dict[int, HalfMatch]:
    space = result_a.space
    stack = np.stack(result_b.references)
    full = np.ones_like(stack)
    candidates = []
    for label_a, reference_a in enumerate(result_a.references):
        aligned = align_to_reference(stack, full, reference_a, space, threads=threads)
        for label_b in range(len(result_b.references)):
            candidates.append((float(aligned.cc[label_b]), label_a, label_b, space.rotations[aligned.rotation_index[label_b]], aligned.shift[label_b].astype(float)))
    bars = pair_thresholds(np.concatenate([result_a.ceilings, result_b.ceilings]), merge_fraction)
    matches: dict[int, HalfMatch] = {}
    used_a: set[int] = set()
    for similarity, label_a, label_b, rotation, shift in sorted(candidates, key=lambda c: -c[0]):
        if label_b in matches or label_a in used_a:
            continue
        if similarity >= bars[label_a, len(result_a.ceilings) + label_b]:
            matches[label_b] = HalfMatch(label_a, similarity, rotation, shift)
            used_a.add(label_a)
    log.debug(f'Matched {len(matches)} of {len(result_b.references)} half-B cluster(s) to half A by reference similarity')
    return matches

# express_in_frame_of_a: half-B poses re-expressed against matched half-A reference, so both halves' averages sit in one frame
def express_in_frame_of_a(result_b: PolishResult, matches: dict[int, HalfMatch]) -> tuple[np.ndarray, np.ndarray]:
    rotations, shifts = result_b.rotations.copy(), result_b.shifts.astype(float)
    for label_b, match in matches.items():
        chosen = result_b.labels == label_b
        shifts[chosen] = np.einsum('nij,j->ni', rotations[chosen], match.shift) + shifts[chosen]
        rotations[chosen] = rotations[chosen] @ match.rotation
    return rotations, shifts

# run_polish_by_half: polish on each half on its own then match half-B to half-A clusters
def run_polish_by_half(
    store: ParticleStore,
    indices: dict[str, list[int]],
    embeddings: dict[str, np.ndarray],
    seed_labels: dict[str, np.ndarray],
    config: PolishConfig,
    *,
    checkpoint_dir: Path | None = None,
) -> tuple[dict[str, PolishResult], dict[int, HalfMatch]]:
    results = {}
    for half in ('A', 'B'):
        log.progress(f'Polish step for half {half}: {len(indices[half])} particle(s)')
        results[half] = run_polish(store.take(indices[half]), embeddings[half], seed_labels[half], config, checkpoint_dir=None if checkpoint_dir is None else checkpoint_dir / half)
    return results, match_halves(results['A'], results['B'], config.merge_fraction, threads=config.threads)
