'''
STAMP: polish step, seed references, reassignment, merging and null-based rejection
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.merge import half_split_ceilings, merge_groups, pair_thresholds, pairwise_similarity
from stamp.classify.null import null_bar, null_threshold
from stamp.classify.references import Assignment, hard_assign, rebuild_references, score_against_refs, seed_reference
from stamp.classify.search import SearchSpace, search_space
from stamp.classify.store import ParticleStore
from stamp.schemas.particles import OrientationPrior
from stamp.utils.checkpoint import atomic_save_npz
from stamp.utils.errors import StampPipelineError
from stamp.utils.log import log

# PolishConfig: tunables of the polish step
@dataclass(frozen=True)
class PolishConfig:
    prior: OrientationPrior
    angular_step_degrees: float = 10.0
    n_rotations: int = 3000     # SO(3) candidates for the none prior
    shift_range_voxels: int = 3
    n_polish_rounds: int = 2
    min_seed_size: int = 20     # smallest seed cluster that gets a reference
    merge_fraction: float = 1.0     # fraction of the higher half-split ceiling two references must reach to merge
    null_sigmas: float = 3.0    # a reference must beat the phase-randomised null agreement by this many null standard deviations
    reject_quantile: float = 0.95   # null quantile of the best correlation below which a particle is rejected
    seed: int = 0
    threads: int = 1

    def __post_init__(self) -> None:
        if self.n_polish_rounds < 1:
            raise ValueError('n_polish_rounds must be at least 1')
        if not 0.0 < self.reject_quantile < 1.0:
            raise ValueError('reject_quantile must be between 0 and 1')

# PolishResult: per-particle outcome of polish
@dataclass
class PolishResult:
    labels: np.ndarray      # (n,), -1 for rejected particles, else 0..K-1
    cc: np.ndarray          # (n,), correlation with the assigned reference
    margin: np.ndarray      # (n,), best minus second-best correlation
    rotations: np.ndarray   # (n, 3, 3), pose against the assigned reference
    shifts: np.ndarray      # (n, 3), working voxels
    polarity: np.ndarray    # (n,), -1 where the pose contains the filament flip
    references: list[np.ndarray]
    space: SearchSpace

# run_polish: refine seed clusters into references, polish them against all particles and reject what matches nothing better than noise; each stage is checkpointed in checkpoint_dir when given
def run_polish(store: ParticleStore, embedding: np.ndarray, seed_labels: np.ndarray, config: PolishConfig, *, checkpoint_dir: Path | None = None) -> PolishResult:
    box_voxels = store.volumes.box_voxels
    space = search_space(config.prior, box_voxels, angular_step_degrees=config.angular_step_degrees, n_rotations=config.n_rotations, shift_range_voxels=config.shift_range_voxels, seed=config.seed)
    references, first_round = _resume(checkpoint_dir, seed_labels, config)
    if references is None:
        seeds = [k for k in np.unique(seed_labels) if k >= 0 and (seed_labels == k).sum() >= config.min_seed_size]
        if not seeds:
            raise StampPipelineError(f'no seed cluster has at least {config.min_seed_size} particles')
        log.progress(f'Building {len(seeds)} seed reference(s)')
        references = [seed_reference(store, embedding, np.where(seed_labels == k)[0], space, threads=config.threads) for k in seeds]
        _save(checkpoint_dir, 'seeds', seed_labels, references)
    for round_number in range(first_round, config.n_polish_rounds):
        log.progress(f'Polish round {round_number + 1}/{config.n_polish_rounds}: {len(references)} reference(s)')
        references = _polish_round(store, references, space, config)
        _save(checkpoint_dir, f'round{round_number}', seed_labels, references)
    assignment = hard_assign(score_against_refs(store, references, space, threads=config.threads), space)
    threshold = null_threshold(store, references, space, quantile=config.reject_quantile, seed=config.seed, threads=config.threads)
    log.debug(f'Rejection threshold from the null: {threshold:.3f}')
    return _finalise(assignment, references, threshold, space)

# _polish_round: reassign every particle, rebuild references, drop those that do not beat the null and merge those that are the same structure
def _polish_round(store: ParticleStore, references: list[np.ndarray], space: SearchSpace, config: PolishConfig) -> list[np.ndarray]:
    box_voxels = store.volumes.box_voxels
    assignment = hard_assign(score_against_refs(store, references, space, threads=config.threads), space)
    ceilings = half_split_ceilings(store.batches(), assignment.labels, assignment.rotations, assignment.shifts, space, len(references), box_voxels, seed=config.seed)
    rebuilt = rebuild_references(store, assignment.labels, assignment.rotations, assignment.shifts, len(references))
    counts = np.bincount(assignment.labels, minlength=len(references))
    bar = null_bar(store, references[int(counts.argmax())], space, max(int(np.median(counts[counts > 0])), 8), sigmas=config.null_sigmas, seed=config.seed, threads=config.threads)
    keep = [k for k, reference in enumerate(rebuilt) if reference is not None and ceilings[k] >= bar]
    log.debug(f'Null bar {bar:.3f}: kept {len(keep)} of {len(references)} reference(s)')
    if not keep:
        raise StampPipelineError('no reference beat the phase-randomised null, the particles share no consistent structure at this resolution')
    kept = [rebuilt[k] for k in keep]
    if len(kept) == 1:
        return kept
    similarity, poses = pairwise_similarity(kept, space, threads=config.threads)
    groups = merge_groups(similarity, pair_thresholds(ceilings[keep], config.merge_fraction))
    return _merge_references(store, assignment, keep, kept, groups, poses, space)

# _merge_references: one reference per merge group, built from every member at its pose composed with the pose of its old reference against the group's largest reference
def _merge_references(store: ParticleStore, assignment: Assignment, keep: list[int], kept: list[np.ndarray], groups: np.ndarray, poses, space: SearchSpace) -> list[np.ndarray]:
    labels = np.full(len(store), -1)
    rotations, shifts = assignment.rotations.copy(), assignment.shifts.astype(float)
    sizes = np.array([(assignment.labels == k).sum() for k in keep])
    for group in np.unique(groups):
        members = np.where(groups == group)[0]
        anchor = members[np.argmax(sizes[members])]
        for position in members:
            chosen = assignment.labels == keep[position]
            labels[chosen] = group
            if position != anchor:
                relative_rotation, relative_shift = poses[anchor][position]    # old reference onto the anchor
                shifts[chosen] = np.einsum('nij,j->ni', rotations[chosen], relative_shift) + shifts[chosen]
                rotations[chosen] = rotations[chosen] @ relative_rotation
    return [reference for reference in rebuild_references(store, labels, rotations, shifts, len(np.unique(groups))) if reference is not None]

# _finalise: reject particles below the null threshold and renumber the surviving references
def _finalise(assignment: Assignment, references: list[np.ndarray], threshold: float, space: SearchSpace) -> PolishResult:
    labels = np.where(assignment.cc >= threshold, assignment.labels, -1)
    used = np.unique(labels[labels >= 0])
    renumbered = np.full_like(labels, -1)
    for new, old in enumerate(used):
        renumbered[labels == old] = new
    flipped = (assignment.rotations[:, 2, 2] < 0) & (space.prior is OrientationPrior.AXIS)  # a filament pose that sends +z to -z carries the polarity flip, other priors have no polarity
    return PolishResult(renumbered, assignment.cc, assignment.margin, assignment.rotations, assignment.shifts, np.where(flipped, -1, 1).astype(np.int8), [references[k] for k in used], space)

# _save: checkpoint the references after a stage together with the seeds they derive from
def _save(checkpoint_dir: Path | None, stage: str, seed_labels: np.ndarray, references: list[np.ndarray]) -> None:
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_save_npz(checkpoint_dir / f'{stage}.npz', seed_labels=seed_labels, references=np.stack(references))

# _resume: references of the latest finished stage and the next round to run, or (None, 0) when nothing usable is saved
def _resume(checkpoint_dir: Path | None, seed_labels: np.ndarray, config: PolishConfig) -> tuple[list[np.ndarray] | None, int]:
    if checkpoint_dir is None:
        return None, 0
    for round_number in range(config.n_polish_rounds - 1, -2, -1):
        path = checkpoint_dir / ('seeds.npz' if round_number < 0 else f'round{round_number}.npz')
        try:
            with np.load(path) as saved:
                if np.array_equal(saved['seed_labels'], seed_labels):
                    log.info(f'Resuming polish step after {path.stem}')
                    return list(saved['references']), round_number + 1
        except (OSError, ValueError, KeyError):
            continue
    return None, 0
