'''
STAMP: unit tests for particle set filter state
'''

# Import external dependencies
import json, mrcfile, numpy as np, pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.filter.state import load_filter_state
from stamp.schemas.particles import Particle, ParticleSet
from stamp.utils.errors import StampPipelineError

# _write_particle_set: write a minimal particle_set.json with two tomograms
def _write_particle_set(path: Path) -> None:
    particles = [
        Particle(particle_id='p0', tomogram_id='tomo000', position=(1.0, 1.0, 1.0), source_picker='mocked-particle', half_set = 'A'),
        Particle(particle_id='p1', tomogram_id='tomo000', position=(2.0, 2.0, 2.0), source_picker='mocked-particle', half_set='A'),
        Particle(particle_id='p2', tomogram_id='tomo001', position=(3.0, 3.0, 3.0), source_picker='mocked-particle', half_set='B'),
    ]
    path.write_text(ParticleSet(particles=particles, consensus_rule='union', contributing_pickers=['mocked-particle']).model_dump_json())

# TestFilterState: tests for FilterState's accept/reject logic
class TestFilterState:
    # check toggle flips a single particle's rejected state
    def test_toggle(self, tmp_path: Path) -> None:
        particle_set_path = tmp_path / 'particle_set.json'
        _write_particle_set(particle_set_path)
        state = load_filter_state(particle_set_path, None, None, output_dir=tmp_path)
        state.toggle('p0')
        assert 'p0' in state.rejected
        state.toggle('p0')
        assert 'p0' not in state.rejected

    # check reject_all only rejects particles in the given tomogram
    def test_reject_all_scoped_to_tomogram(self, tmp_path: Path) -> None:
        particle_set_path = tmp_path / 'particle_set.json'
        _write_particle_set(particle_set_path)
        state = load_filter_state(particle_set_path, None, None, output_dir=tmp_path)
        state.reject_all('tomo000')
        assert state.rejected == {'p0', 'p1'}

    # check reset_tomogram clears only that tomogram's rejections
    def test_reset_tomogram(self, tmp_path: Path) -> None:
        particle_set_path = tmp_path / 'particle_set.json'
        _write_particle_set(particle_set_path)
        state = load_filter_state(particle_set_path, None, None, output_dir=tmp_path)
        state.rejected = {'p0', 'p1', 'p2'}
        state.reset_tomogram('tomo000')
        assert state.rejected == {'p2'}

    # check save writes accepted-only particle_set.filtered.json and rejected_picks.json
    def test_save_writes_both_outputs(self, tmp_path: Path) -> None:
        particle_set_path = tmp_path / 'particle_set.json'
        _write_particle_set(particle_set_path)
        state = load_filter_state(particle_set_path, None, None, output_dir=tmp_path)
        state.reject_all('tomo000')
        filtered_path, rejected_path = state.save(tmp_path)
        filtered = ParticleSet.model_validate(json.loads(filtered_path.read_text()))
        assert {p.particle_id for p in filtered.particles} == {'p2'}
        assert rejected_path.exists()
