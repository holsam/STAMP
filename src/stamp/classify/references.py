'''
STAMP: references for class polishing, scoring particles against them and rebuilding them from members
'''

# Import external dependencies
import hashlib
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# Import internal STAMP objects
from stamp.classify.average import WeightedAverage
from stamp.classify.match import Alignment, align_to_reference
from stamp.classify.search import SearchSpace, rotated_bank
from stamp.classify.store import ParticleStore
from stamp.utils.checkpoint import atomic_save_npz
from stamp.utils.log import log

# Scores: every particle aligned to every reference, first axis is the reference
@dataclass(frozen=True)
class Scores:
    cc: np.ndarray              # (n_refs, n)
    rotation_index: np.ndarray  # (n_refs, n)
    shift: np.ndarray           # (n_refs, n, 3)

# Assignment: each particle's best reference and its pose against that reference
@dataclass(frozen=True)
class Assignment:
    labels: np.ndarray      # (n,), reference index
    cc: np.ndarray          # (n,), correlation with the assigned reference
    margin: np.ndarray      # (n,), best minus second-best correlation across references (the correlation itself with one reference)
    rotations: np.ndarray   # (n, 3, 3)
    shifts: np.ndarray      # (n, 3), working voxels

# _load_partial_alignment: arrays and done mask saved for this exact reference and store size, else None
def _load_partial_alignment(path: Path | None, digest: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    try:
        with np.load(path) as saved:
            if str(saved['digest']) != digest or len(saved['done']) != n:
                return None
            return saved['rotation_index'], saved['shift'], saved['cc'], saved['done']
    except (OSError, ValueError, KeyError):
        return None

# _save_partial_alignment: checkpoint the arrays and done mask so a restart skips the finished batches
def _save_partial_alignment(path: Path | None, digest: str, rotation_index: np.ndarray, shift: np.ndarray, cc: np.ndarray, done: np.ndarray) -> None:
    if path is not None:
        atomic_save_npz(path, digest=np.array(digest), rotation_index=rotation_index, shift=shift, cc=cc, done=done)

# align_store: align every particle of a store to one reference, streaming in batches
def align_store(
    store: ParticleStore,
    reference: np.ndarray,
    space: SearchSpace,
    *,
    threads: int = 1,
    batch_size: int = 256,
    checkpoint: Path | None = None,
    label: str = 'alignment',
) -> Alignment:
    n = len(store)
    digest = hashlib.sha1(np.ascontiguousarray(reference).tobytes()).hexdigest()     # ties the file to this exact reference
    resumed = _load_partial_alignment(checkpoint, digest, n)
    if resumed is None:
        rotation_index, shift, cc, done = np.empty(n, dtype=np.int64), np.zeros((n, 3), dtype=np.int64), np.empty(n, dtype=np.float32), np.zeros(n, dtype=bool)
    else:
        rotation_index, shift, cc, done = resumed
        if done.any():
            log.info(f'Resuming {label}: {int(done.sum())}/{n} particles already aligned')
    pending = [chunk for chunk in store.volumes.batch_indices(batch_size) if not done[chunk].all()]
    if pending:
        bank = rotated_bank(reference, space.rotations, threads=threads)
        report_every = max(1, len(pending) // 10)
        for number, chunk in enumerate(pending, 1):
            volumes, wedges = store.load(chunk)
            result = align_to_reference(volumes, wedges, reference, space, threads=threads, bank=bank)
            rotation_index[chunk], shift[chunk], cc[chunk] = result.rotation_index, result.shift, result.cc
            done[chunk] = True
            if number % report_every == 0 or number == len(pending):
                log.progress(f'{label}: {int(done.sum())}/{n} complete')
                _save_partial_alignment(checkpoint, digest, rotation_index, shift, cc, done)
    return Alignment(rotation_index, shift, cc)

# score_against_refs: alignment of every particle to every reference
def score_against_refs(store: ParticleStore, references: list[np.ndarray], space: SearchSpace, *, threads: int = 1, checkpoint_dir: Path | None = None, label: str = 'scoring') -> Scores:
    results = [align_store(store, reference, space, threads=threads, checkpoint=None if checkpoint_dir is None else checkpoint_dir / f'ref{k:02d}.npz', label=f'{label} reference {k + 1}/{len(references)}') for k, reference in enumerate(references)]
    return Scores(np.stack([r.cc for r in results]), np.stack([r.rotation_index for r in results]), np.stack([r.shift for r in results]))

# hard_assign: best reference per particle, with its pose
def hard_assign(scores: Scores, space: SearchSpace) -> Assignment:
    labels = scores.cc.argmax(axis=0)
    columns = np.arange(scores.cc.shape[1])
    ordered = np.sort(scores.cc, axis=0)
    margin = ordered[-1] - ordered[-2] if len(ordered) > 1 else ordered[-1]
    return Assignment(labels, scores.cc[labels, columns], margin, space.rotations[scores.rotation_index[labels, columns]], scores.shift[labels, columns, :])

# rebuild_references: coverage-corrected average of each label's members at their poses
def rebuild_references(store: ParticleStore, labels: np.ndarray, rotations: np.ndarray, shifts: np.ndarray, n_refs: int, *, min_members: int = 3, batch_size: int = 256, threads: int = 1) -> list[np.ndarray | None]:
    accumulators = {k: WeightedAverage(store.volumes.box_voxels) for k in range(n_refs) if (labels == k).sum() >= min_members}
    for indices, volumes, wedges in store.batches(batch_size):
        indices = np.array(indices)
        for k, accumulator in accumulators.items():
            chosen = labels[indices] == k
            if chosen.any():
                accumulator.add(volumes[chosen], wedges[chosen], rotations[indices[chosen]], shifts[indices[chosen]], threads=threads)
    return [accumulators[k].result() if k in accumulators else None for k in range(n_refs)]

# refine_reference: grow a reference from a single seed particle by iterated alignment of the members and re-averaging
def refine_reference(members: ParticleStore, seed: np.ndarray, space: SearchSpace, *, n_iter: int = 3, threads: int = 1) -> np.ndarray:
    reference = seed
    for iteration in range(n_iter):
        alignment = align_store(members, reference, space, threads=threads, label=f'seed refinement {iteration + 1}/{n_iter}')
        reference = rebuild_references(members, np.zeros(len(members), dtype=int), space.rotations[alignment.rotation_index], alignment.shift, 1, min_members=1, threads=threads)[0]
    return reference

# seed_reference: reference for a seed cluster, started from the member closest to the cluster's mean in feature space
def seed_reference(store: ParticleStore, embedding: np.ndarray, members: np.ndarray, space: SearchSpace, *, threads: int = 1) -> np.ndarray:
    medoid = int(members[np.argmin(np.linalg.norm(embedding[members] - embedding[members].mean(axis=0), axis=1))])
    group = store.take(members.tolist())
    return refine_reference(group, store.load([medoid])[0][0], space, threads=threads)
