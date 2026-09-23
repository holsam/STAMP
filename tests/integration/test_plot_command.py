'''
STAMP: integration tests for `stamp plot`
'''

# Import external dependencies
import json, mrcfile, numpy as np
from pathlib import Path
from typer.testing import CliRunner

# Import internal STAMP objects
from stamp.cli.cli import stamp_app
from stamp.schemas.particles import Particle, ParticleSet

# Initialise runner
runner = CliRunner()

# _write_particle_set: write a minimal particle_set.json with two tomograms, one particle each
def _write_particle_set(path: Path) -> ParticleSet:
    particles = [
        Particle(particle_id='p0', tomogram_id='tomo000', position=(30.0, 30.0, 30.0), source_picker='mocked-particle', half_set='A'),
        Particle(particle_id='p1', tomogram_id='tomo001', position=(30.0, 30.0, 30.0), source_picker='mocked-particle', half_set='B'),
    ]
    particle_set = ParticleSet(particles=particles, consensus_rule='union', contributing_pickers=['mocked-particle'])
    path.write_text(particle_set.model_dump_json())
    return particle_set

# _write_raw: write a blank raw tomogram MRC
def _write_raw(raw_dir: Path, tomogram_id: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(raw_dir / f'{tomogram_id}.mrc', overwrite=True) as mrc:
        mrc.set_data(np.zeros((60, 60, 60), dtype=np.float32))

# TestPlotPickCommand: tests for `stamp plot pick`
class TestPlotPickCommand:
    # check plotting all tomograms writes one plot per tomogram
    def test_plots_all_tomograms(self, tmp_path: Path) -> None:
        particle_set_path = tmp_path / 'particle_set.json'
        _write_particle_set(particle_set_path)
        raw_dir = tmp_path / 'raw'
        _write_raw(raw_dir, 'tomo000')
        _write_raw(raw_dir, 'tomo001')
        out_dir = tmp_path / 'plots'
        result = runner.invoke(stamp_app, ['plot', 'pick', str(particle_set_path), '--raw-dir', str(raw_dir), '--out-dir', str(tmp_path), '--pick-plot-style', 'none', '--no-pick-zstack-movie', '--plot-format', 'png'])
        assert result.exit_code == 0, result.output
        assert list(out_dir.glob('*.png'))

    # check --tomogram-ids restricts plotting to the named tomogram(s)
    def test_restricts_by_tomogram_ids(self, tmp_path: Path) -> None:
        particle_set_path = tmp_path / 'particle_set.json'
        _write_particle_set(particle_set_path)
        raw_dir = tmp_path / 'raw'
        _write_raw(raw_dir, 'tomo000')
        _write_raw(raw_dir, 'tomo001')
        out_dir = tmp_path / 'plots'
        result = runner.invoke(stamp_app, ['plot', 'pick', str(particle_set_path), '--raw-dir', str(raw_dir), '--out-dir', str(tmp_path), '--pick-plot-style', 'none', '--no-pick-zstack-movie', '--plot-format', 'png', '--tomogram-ids', 'tomo000'])
        assert result.exit_code == 0, result.output
        assert 'Plotting 1 of 2 tomogram(s)' in result.output

    # check an unknown tomogram ID errors
    def test_unknown_tomogram_id_errors(self, tmp_path: Path) -> None:
        particle_set_path = tmp_path / 'particle_set.json'
        _write_particle_set(particle_set_path)
        raw_dir = tmp_path / 'raw'
        _write_raw(raw_dir, 'tomo000')
        _write_raw(raw_dir, 'tomo001')
        result = runner.invoke(stamp_app, ['plot', 'pick', str(particle_set_path), '--raw-dir', str(raw_dir), '--out-dir', str(tmp_path), '--tomogram-ids', 'not-a-tomogram'])
        assert result.exit_code != 0

    # check --tomogram-ids and --n-tomograms are mutually exclusive
    def test_tomogram_ids_and_n_tomograms_mutually_exclusive(self, tmp_path: Path) -> None:
        particle_set_path = tmp_path / 'particle_set.json'
        _write_particle_set(particle_set_path)
        raw_dir = tmp_path / 'raw'
        _write_raw(raw_dir, 'tomo000')
        result = runner.invoke(stamp_app, ['plot', 'pick', str(particle_set_path), '--raw-dir', str(raw_dir), '--tomogram-ids', 'tomo000', '--n-tomograms', '1'])
        assert result.exit_code != 0
