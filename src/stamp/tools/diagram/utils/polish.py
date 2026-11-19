'''
STAMP: 2D mocked polish step, seed references, reassignment, merging and null-based rejection
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

# Import internal STAMP objects
from stamp.tools.diagram.utils.align import Alignment, align_stack, align_to_references, apply_pose, bin_stack, weighted_average
from stamp.tools.diagram.utils.search import SearchSpace

# Stacks: one set of particles at full and working resolution with their wedge masks
@dataclass
class Stacks:
    full: np.ndarray
    masks: np.ndarray
    binned: np.ndarray
    masks_binned: np.ndarray
    factor: int

    def take(self, index: np.ndarray) -> 'Stacks':
        return Stacks(self.full[index], self.masks[index], self.binned[index], self.masks_binned[index], self.factor)

# PolishConfig: tunable settings for polish step
@dataclass(frozen=True)
class PolishConfig:
    n_rounds: int = 2
    min_seed_size: int = 12
    merge_fraction: float = 1.0
    null_sigmas: float = 3.0
    reject_quantile: float = 0.95
    seed: int = 0

# PolishResult: per-particle outcome of polish, plus what diagram draws
@dataclass
class PolishResult:
    labels: np.ndarray      # (n,), -1 for rejected particles, else 0..K-1
    cc: np.ndarray          # (n,) correlation with assigned reference
    margin: np.ndarray      # (n,) best minus second-best correlation
    angle: np.ndarray       # (n,) degrees, rotation of the reference that matches particle
    shift: np.ndarray       # (n, 2) full-resolution pixels
    references: list[np.ndarray]
    seed_references: list[np.ndarray]
    ceilings: np.ndarray    # (K_before_merge,) half-split agreement
    similarity: np.ndarray  # (K_before_merge, K_before_merge) reference agreement used for merging
    merged: list[tuple[int, int]]
    null_best: np.ndarray   # (n,) best correlation of each particle against phase-randomised references
    particle_bar: float     # null quantile below which a particle is rejected
    reference_bar: float    # null bar a reference's median correlation must clear
    reference_cc: list[float]

# _ones: wedge masks that observe everything, for comparing two averages
def _ones(n: int, size: int) -> np.ndarray:
    return np.ones((n, size, size), dtype=np.float32)

# _average: wedge-weighted average of particles at their poses, full resolution
def _average(stacks: Stacks, angle: np.ndarray, shift: np.ndarray) -> np.ndarray:
    return weighted_average(*align_stack(stacks.full, stacks.masks, angle, shift, stacks.factor))

# _seed_reference: align a cluster's members to the member nearest its centroid and average, twice; mirrors classify/references.seed_reference
def _seed_reference(stacks: Stacks, embedding: np.ndarray, space: SearchSpace) -> np.ndarray:
    medoid = int(np.argmin(np.linalg.norm(embedding - embedding.mean(axis=0), axis=1)))
    reference = stacks.full[medoid]
    for _ in range(2):
        alignment = align_to_references(stacks.binned, stacks.masks_binned, bin_stack(reference[None], stacks.factor), space)
        reference = _average(stacks, alignment.angle[:, 0], alignment.shift[:, 0])
    return reference

# _assign: align every particle to every reference and assign each to its best
def _assign(stacks: Stacks, references: list[np.ndarray], space: SearchSpace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alignment = align_to_references(stacks.binned, stacks.masks_binned, bin_stack(np.array(references), stacks.factor), space)
    rows = np.arange(len(stacks.full))
    labels = alignment.cc.argmax(axis=1)
    ordered = np.sort(alignment.cc, axis=1)
    margin = ordered[:, -1] - ordered[:, -2] if ordered.shape[1] > 1 else ordered[:, -1]
    return labels, alignment.cc[rows, labels], margin, alignment.angle[rows, labels], alignment.shift[rows, labels]

# _rebuild: new references from particles assigned to each, a reference with fewer than three members is kept as it was
def _rebuild(stacks: Stacks, references: list[np.ndarray], labels: np.ndarray, angle: np.ndarray, shift: np.ndarray) -> list[np.ndarray]:
    rebuilt = []
    for k, reference in enumerate(references):
        members = np.where(labels == k)[0]
        rebuilt.append(_average(stacks.take(members), angle[members], shift[members]) if len(members) >= 3 else reference)
    return rebuilt

# _pair_cc: best agreement of one average with another over search space, with pose to gets there
def _pair_cc(first: np.ndarray, second: np.ndarray, space: SearchSpace, factor: int) -> Alignment:
    return align_to_references(bin_stack(first[None], factor), _ones(1, first.shape[-1] // factor), bin_stack(second[None], factor), space)

# _ceiling: half-split agreement of a reference, how well two independent averages of its members agree
def _ceiling(stacks: Stacks, angle: np.ndarray, shift: np.ndarray, space: SearchSpace, rng: np.random.Generator) -> float:
    order = rng.permutation(len(stacks.full))
    half_a, half_b = order[: len(order) // 2], order[len(order) // 2:]
    average_a = _average(stacks.take(half_a), angle[half_a], shift[half_a])
    average_b = _average(stacks.take(half_b), angle[half_b], shift[half_b])
    return float(_pair_cc(average_b, average_a, space, stacks.factor).cc[0, 0])

# _phase_randomise: keep amplitude spectrum, replace phases with those of white noise
def _phase_randomise(reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise_phase = np.angle(np.fft.fft2(rng.normal(size=reference.shape)))
    return np.fft.ifft2(np.abs(np.fft.fft2(reference)) * np.exp(1j * noise_phase)).real

# run_polish: refine seed clusters into references, polish them against all particles, merge references that agree as well as a reference agrees with itself, and reject what matches nothing better than noise
def run_polish(stacks: Stacks, embedding: np.ndarray, seed_labels: np.ndarray, space: SearchSpace, config: PolishConfig) -> PolishResult:
    rng = np.random.default_rng(config.seed)
    seeds = [k for k in np.unique(seed_labels) if k >= 0 and (seed_labels == k).sum() >= config.min_seed_size]
    references = [_seed_reference(stacks.take(np.where(seed_labels == k)[0]), embedding[seed_labels == k], space) for k in seeds]
    seed_references = list(references)
    for _ in range(config.n_rounds):
        labels, cc, margin, angle, shift = _assign(stacks, references, space)
        references = _rebuild(stacks, references, labels, angle, shift)

    # merge: references whose mutual agreement reaches higher half-split ceiling are one class
    ceilings = np.array([_ceiling(stacks.take(np.where(labels == k)[0]), angle[labels == k], shift[labels == k], space, rng) if (labels == k).sum() >= 6 else 0.0 for k in range(len(references))])
    similarity = np.array([[_pair_cc(references[i], references[j], space, stacks.factor).cc[0, 0] if i != j else 1.0 for j in range(len(references))] for i in range(len(references))])
    merged, absorbed = [], set()
    for i in range(len(references)):
        for j in range(i + 1, len(references)):
            if i not in absorbed and j not in absorbed and similarity[i, j] >= config.merge_fraction * max(ceilings[i], ceilings[j]):
                pose = _pair_cc(references[j], references[i], space, stacks.factor)
                counts = np.array([(labels == i).sum(), (labels == j).sum()])
                moved = apply_pose(references[j], float(pose.angle[0, 0]), tuple(pose.shift[0, 0] * stacks.factor))
                references[i] = (counts[0] * references[i] + counts[1] * moved) / counts.sum()
                merged.append((i, j))
                absorbed.add(j)
    references = [reference for k, reference in enumerate(references) if k not in absorbed]
    labels, cc, margin, angle, shift = _assign(stacks, references, space)
    references = _rebuild(stacks, references, labels, angle, shift)
    labels, cc, margin, angle, shift = _assign(stacks, references, space)

    # null: correlation with phase-randomised references sets bars a real match has to clear
    null_refs = [_phase_randomise(reference, rng) for reference in references]
    null_best = align_to_references(stacks.binned, stacks.masks_binned, bin_stack(np.array(null_refs), stacks.factor), space).cc.max(axis=1)
    particle_bar = float(np.quantile(null_best, config.reject_quantile))
    reference_bar = float(null_best.mean() + config.null_sigmas * null_best.std())
    labels = np.where(cc >= particle_bar, labels, -1)
    reference_cc = [float(np.median(cc[labels == k])) if (labels == k).any() else 0.0 for k in range(len(references))]
    for k, value in enumerate(reference_cc):
        if value < reference_bar:
            labels[labels == k] = -1
    kept = [k for k in range(len(references)) if (labels == k).any()]
    renumber = {old: new for new, old in enumerate(kept)}
    labels = np.array([renumber.get(label, -1) for label in labels])
    return PolishResult(labels, cc, margin, angle, shift * stacks.factor, [references[k] for k in kept], seed_references, ceilings, similarity, merged, null_best, particle_bar, reference_bar, [reference_cc[k] for k in kept])

# match_halves: pair each half-B reference with a half-A reference by agreement (one to one), returning B index, A index and pose to brings B into frame of A
def match_halves(references_a: list[np.ndarray], references_b: list[np.ndarray], space: SearchSpace, factor: int) -> list[tuple[int, int, float, np.ndarray, float]]:
    alignment = align_to_references(bin_stack(np.array(references_b), factor), _ones(len(references_b), references_b[0].shape[-1] // factor), bin_stack(np.array(references_a), factor), space)
    rows, cols = linear_sum_assignment(-alignment.cc)
    return [(int(b), int(a), float(alignment.angle[b, a]), alignment.shift[b, a] * factor, float(alignment.cc[b, a])) for b, a in zip(rows, cols)]
