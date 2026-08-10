'''
STAMP: unit tests for half set utility functions
'''

# Import external dependencies
import pytest

# Import internal functions and schema
from stamp.utils.halfset import assign_half_sets, split_by_half_set, validate_single_half_set
from stamp.schemas.particles import HalfSet, Particle

# _particle: returns a valid Particle instance
def _particle(particle_id: str, half_set: HalfSet) -> Particle:
    return Particle(
        particle_id=particle_id,
        tomogram_id='tomo000',
        position=(0.0, 0.0, 0.0),
        source_picker='example-picker',
        half_set=half_set,
    )

class TestAssignHalfSet:
    def test_same_seed_and_ids_give_identical_split(self) -> None:
        '''The same seed and ids should give an identical split'''
        ids = [f'p{i:03d}' for i in range(50)]
        first = assign_half_sets(ids, seed=7)
        second = assign_half_sets(ids, seed=7)
        assert first == second

    def test_split_is_independent_of_input_order(self) -> None:
        '''Particles should be split irrespective of input order'''
        ids = [f'p{i:03d}' for i in range(50)]
        shuffled_ids = list(reversed(ids))
        assert assign_half_sets(ids, seed=7) == assign_half_sets(shuffled_ids, seed=7)

    def test_different_seeds_give_different_splits(self) -> None:
        '''A different seed should give a different split'''
        ids = [f'p{i:03d}' for i in range(50)]
        first = assign_half_sets(ids, seed=1)
        second = assign_half_sets(ids, seed=2)
        assert first != second

    def test_split_is_balanced_for_even_count(self) -> None:
        '''For an even number of particles, each half set should have the same number of particles'''
        ids = [f'p{i:03d}' for i in range(40)]
        split = assign_half_sets(ids, seed=3)
        counts = {half_set: list(split.values()).count(half_set) for half_set in HalfSet}
        assert counts[HalfSet.A] == counts[HalfSet.B] == 20


    def test_split_gives_extra_particle_to_a_for_odd_count(self) -> None:
        '''For an odd number of particles, half set A should have the extra particle'''
        ids = [f'p{i:03d}' for i in range(41)]
        split = assign_half_sets(ids, seed=3)
        counts = {half_set: list(split.values()).count(half_set) for half_set in HalfSet}
        assert counts[HalfSet.A] == 21
        assert counts[HalfSet.B] == 20

    def test_assign_half_sets_rejects_empty_list(self) -> None:
        '''An empty particle list should raise an error'''
        with pytest.raises(ValueError, match='empty'):
            assign_half_sets([], seed=0)

    def test_assign_half_sets_rejects_duplicate_ids(self) -> None:
        '''A particle list with duplicate ids should raise an error'''
        with pytest.raises(ValueError, match='duplicates'):
            assign_half_sets(['p001', 'p001'], seed=0)

class TestValidateHalfSet:
    def test_validate_single_half_set_accepts_uniform_list(self) -> None:
        '''A list of particles in the same half set should return the corresponding half set'''
        particles = [_particle('p001', HalfSet.A), _particle('p002', HalfSet.A)]
        assert validate_single_half_set(particles) == HalfSet.A

    def test_validate_single_half_set_rejects_mixed_list(self) -> None:
        '''A list of particles in mixed half sets should raise an error'''
        particles = [_particle('p001', HalfSet.A), _particle('p002', HalfSet.B)]
        with pytest.raises(ValueError, match='mixes half-sets'):
            validate_single_half_set(particles)

    def test_validate_single_half_set_rejects_empty_list(self) -> None:
        '''An empty particle list should raise an error'''
        with pytest.raises(ValueError, match='empty'):
            validate_single_half_set([])

class TestSplitHalfSet:
    '''A list of particles from different half sets should be split into separate lists'''
    def test_split_by_half_set(self) -> None:
        particles = [
            _particle('p001', HalfSet.A),
            _particle('p002', HalfSet.B),
            _particle('p003', HalfSet.A),
        ]
        half_a, half_b = split_by_half_set(particles)
        assert {p.particle_id for p in half_a} == {'p001', 'p003'}
        assert {p.particle_id for p in half_b} == {'p002'}