'''
STAMP: half-set assignment and validation
'''

# Import external dependencies
import random

# Import STAMP schema
from stamp.schemas.particles import HalfSet, Particle

# assign_half_sets: returns a dictionary of strings and a deterministically-assigned half-set
def assign_half_sets(particle_ids: list[str], seed: int) -> dict[str, HalfSet]:
    if not particle_ids:
        raise ValueError('Cannot assign half-sets to an empty list of particle IDs')
    if len(set(particle_ids)) != len(particle_ids):
        raise ValueError('particle_ids contains duplicates; half-set assignment requires unique IDs')
    ordered_ids = sorted(particle_ids)
    random.Random(seed).shuffle(ordered_ids)
    midpoint = (len(ordered_ids) + 1) // 2  # A gets the extra particle if the count is odd
    half_a_ids = set(ordered_ids[:midpoint])
    return {particle_id: (HalfSet.A if particle_id in half_a_ids else HalfSet.B) for particle_id in particle_ids}

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