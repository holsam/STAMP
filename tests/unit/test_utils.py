'''
STAMP: unit tests for utilities
'''

# Import external dependencies
import pytest, subprocess, tomllib

# Import internal functions and schema
from stamp.utils.halfset import assign_half_sets, split_by_half_set, validate_single_half_set
from stamp.schemas.particles import HalfSet, Particle
from stamp.utils import io as io_utils
from stamp.schemas.provenance import ProvenanceSidecar

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

class TestIo:
    def test_resolve_commit_reads_git_head(self):
        assert io_utils.resolve_stamp_commit() != 'unknown'

    def test_resolve_commit_falls_back_when_git_missing(self, monkeypatch):
        def _no_git(*args, **kwargs):
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, 'run', _no_git)
        monkeypatch.setattr(io_utils, 'version', lambda _name: '9.9.9')
        assert io_utils.resolve_stamp_commit() == 'stamp-9.9.9'

    def test_checksum_is_stable_and_cache_hits(self, tmp_path):
        target = tmp_path / 'a.bin'
        target.write_bytes(b'stamp')
        first = io_utils.checksum_file(target, cache_dir=tmp_path)
        assert (tmp_path / '.stamp_checksums.json').is_file()
        assert io_utils.checksum_file(target, cache_dir=tmp_path) == first

    def test_directory_digest_changes_when_a_file_changes(self, tmp_path):
        directory = tmp_path / 'averages'
        directory.mkdir()
        (directory / 'c00.mrc').write_bytes(b'one')
        before = io_utils.checksum_inputs([('class_averages', directory)])['class_averages']
        (directory / 'c00.mrc').write_bytes(b'two')
        after = io_utils.checksum_inputs([('class_averages', directory)])['class_averages']
        assert before != after

    def test_directory_digest_sees_nested_files(self, tmp_path):
        nested = tmp_path / 'root' / 'a' / 'b'
        nested.mkdir(parents=True)
        (nested / 'x.txt').write_text('one')
        before = io_utils.checksum_inputs([('root', tmp_path / 'root')])['root']
        (nested / 'x.txt').write_text('two')
        after = io_utils.checksum_inputs([('root', tmp_path / 'root')])['root']
        assert before != after

    def test_missing_input_recorded_as_absent(self, tmp_path):        
        assert io_utils.checksum_inputs([('gone', tmp_path / 'nope.mrc')]) == {'gone': 'absent'}

    def test_write_sidecar_round_trips(self, tmp_path):
        an_input = tmp_path / 'particle_set.json'
        an_input.write_text('{}')
        path = io_utils.write_sidecar(
            tmp_path / 'out',
            stage='classify',
            tool='stamp-native-classifier',
            tool_version=None,
            parameters={'method': 'hdbscan'},
            inputs=[('particle_set', an_input)],
        )
        parsed = ProvenanceSidecar.model_validate(tomllib.loads(path.read_text()))
        assert parsed.stamp_commit != 'unknown'
        assert parsed.input_checksums['particle_set']
