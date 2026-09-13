'''
STAMP: half-set assignment and validation
'''

# Import external dependencies
import random

# Import STAMP schema
from stamp.schemas.particles import HalfSet, Particle

# assign_half_sets: deterministic A/B split; particles sharing a group key stay in the same half
def assign_half_sets(particle_ids: list[str], seed: int, *, group_of: dict[str, str] | None = None) -> dict[str, HalfSet]:
    if not particle_ids:
        raise ValueError('Cannot assign half-sets to an empty list of particle IDs')
    if len(set(particle_ids)) != len(particle_ids):
        raise ValueError('particle_ids contains duplicates; half-set assignment requires unique IDs')
    group_of = group_of or {particle_id: particle_id for particle_id in particle_ids}
    groups: dict[str, list[str]] = {}
    for particle_id in sorted(particle_ids):
        groups.setdefault(group_of[particle_id], []).append(particle_id)
    ordered_groups = sorted(groups)
    random.Random(seed).shuffle(ordered_groups)
    target = (len(particle_ids) + 1) // 2  # A gets the extra particle if the count is odd
    half_a: set[str] = set()
    for group in ordered_groups:
        if len(half_a) < target:
            half_a.update(groups[group])
    return {particle_id: (HalfSet.A if particle_id in half_a else HalfSet.B) for particle_id in particle_ids}

# validate_single_half_set: confirms each particle in given list belongs to the same half-set (to avoid mixes) 
def validate_single_half_set(particles: list[Particle]) -> HalfSet:
    if not particles:
        raise ValueError('Cannot validate half-set membership of an empty particle list')
    half_sets = {particle.half_set for particle in particles}
    if len(half_sets) > 1:
        raise ValueError(f'Particle list mixes half-sets {sorted(h.value for h in half_sets)}')
    return half_sets.pop()

# split_by_half_set: split a list of Particles into a tuple of lists of Particles by assigned half-set
def split_by_half_set(particles: list[Particle]) -> tuple[list[Particle], list[Particle]]:
    half_a = [particle for particle in particles if particle.half_set == HalfSet.A]
    half_b = [particle for particle in particles if particle.half_set == HalfSet.B]
    return half_a, half_b