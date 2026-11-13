'''
STAMP: label-free null for reference agreement and particle rejection, from phase-randomised particles
'''

# Import external dependencies
import numpy as np

# Import internal STAMP objects
from stamp.classify.match import align_to_reference
from stamp.classify.merge import half_split_ceilings
from stamp.classify.search import SearchSpace
from stamp.classify.store import ParticleStore
from stamp.classify.wedge import apply_wedge
from stamp.utils.phase import phase_randomise

# null_particles: phase-randomised copies of random particles, each limited to its own wedge again so spectrum and wedge stay real
def null_particles(store: ParticleStore, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    volumes, wedges = store.sample(count, rng)
    return np.stack([apply_wedge(phase_randomise(v, rng), w) for v, w in zip(volumes, wedges)]), wedges

# null_threshold: quantile of the best correlation with any reference among null particles
def null_threshold(store: ParticleStore, references: list[np.ndarray], space: SearchSpace, *, quantile: float, sample_size: int = 100, seed: int = 0, threads: int = 1) -> float:
    fake, wedges = null_particles(store, sample_size, np.random.default_rng(seed))
    best = np.max([align_to_reference(fake, wedges, reference, space, threads=threads).cc for reference in references], axis=0)
    return float(np.quantile(best, quantile))

# null_bar: acceptance bar for a reference's half-split agreement, null mean plus sigmas null standard deviations over repeats of size aligned to the reference the same way
def null_bar(store: ParticleStore, reference: np.ndarray, space: SearchSpace, size: int, *, sigmas: float, repeats: int = 8, seed: int = 0, threads: int = 1) -> float:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        fake, wedges = null_particles(store, size, rng)
        alignment = align_to_reference(fake, wedges, reference, space, threads=threads)
        scores.append(half_split_ceilings([(list(range(len(fake))), fake, wedges)], np.zeros(len(fake), dtype=int), space.rotations[alignment.rotation_index], alignment.shift.astype(float), space, 1, fake.shape[1], seed=int(rng.integers(1 << 30)))[0])
    return float(np.mean(scores) + sigmas * np.std(scores, ddof=1))
