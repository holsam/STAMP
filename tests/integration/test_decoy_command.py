'''
STAMP: integration tests of `stamp decoy` for all three methods
'''

# Import external dependencies
import json, mrcfile, numpy as np
from pathlib import Path
from typer.testing import CliRunner

# Import internal STAMP objects
from stamp.cli.cli import stamp_app
from stamp.schemas.particles import ParticleSet
from stamp.utils.errors import StampPipelineError

# Initialise runner
runner = CliRunner()

# _write_pair: write a matched segmentation/tomogram vesicle pair into the given directories
def _write_pair(seg_dir: Path, raw_dir: Path, tomogram_id: str) -> None:
    shape = (60, 60, 60)
    centre = np.array(shape) / 2.0
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1)
    distance = np.linalg.norm(grid - centre, axis=-1)
    segmentation = ((distance > 16.0) & (distance < 20.0)).astype(np.float32)
    tomogram = np.zeros(shape, dtype=np.float32)
    tomogram[segmentation > 0] = -1.0
    tomogram[42:47, 28:33, 28:33] = -8.0

    seg_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(seg_dir / f'{tomogram_id}.mrc', overwrite=True) as mrc:
        mrc.set_data(segmentation)
    with mrcfile.new(raw_dir / f'{tomogram_id}.mrc', overwrite=True) as mrc:
        mrc.set_data(tomogram)

# _run_pick: run `stamp pick` on fresh vesicle pairs and return its paths
def _run_pick(tmp_path: Path, n_tomograms: int = 1) -> tuple[Path, Path, Path]:
    seg_dir, raw_dir, pick_out = tmp_path / 'seg', tmp_path / 'raw', tmp_path / 'stamp' / 'pick'
    for index in range(n_tomograms):
        _write_pair(seg_dir, raw_dir, f'tomo{index:03d}')
    result = runner.invoke(
        stamp_app,
        [
            '-d', str(tmp_path),
            'pick',
            '--seg-dir', str(seg_dir),
            '--raw-dir', str(raw_dir),
            '--out-dir', tmp_path,
            '--voxel-size-a', '10.0',
            '--picker-params', json.dumps({'stamp-native': {'n_mad': 2.5, 'offset_windows_angstrom': [[20.0, 80.0]]}}),
            '--backend', 'local',
            'stamp-native'
        ],
    )
    assert result.exit_code == 0, result.output
    return seg_dir, raw_dir, pick_out / 'particle_set.json'

# _positions: decoy positions from a decoy_particle_set.json, in file order
def _positions(path: Path) -> list[tuple[float, float, float]]:
    return [p.position for p in ParticleSet.model_validate(json.loads(path.read_text())).particles]

# _rejected_surface_args: CLI arguments for a rejected-surface decoy run writing under out_dir
def _rejected_surface_args(tmp_path: Path, out_dir: Path, seg_dir: Path, raw_dir: Path, particle_set_path: Path) -> list[str]:
    return [
        '-d', str(tmp_path),
        'decoy',
        '--method', 'rejected-surface',
        '--real-particle-set', str(particle_set_path),
        '--seg-dir', str(seg_dir),
        '--raw-dir', str(raw_dir),
        '--out-dir', str(out_dir),
        '--voxel-size-a', '10.0',
        '--picker-params', json.dumps({'n_mad': 2.5, 'offset_windows_angstrom': [[20.0, 80.0]]}),
        '--n-decoys-per-tomogram', '20',
        '--seed', '1',
        '--n-processes', '1',
    ]

# _shifted_args: CLI arguments for a shifted decoy run writing under out_dir
def _shifted_args(tmp_path: Path, out_dir: Path, seg_dir: Path, raw_dir: Path, particle_set_path: Path) -> list[str]:
    return [
        '-d', str(tmp_path),
        'decoy',
        '--method', 'shifted',
        '--real-particle-set', str(particle_set_path),
        '--seg-dir', str(seg_dir),
        '--raw-dir', str(raw_dir),
        '--out-dir', str(out_dir),
        '--voxel-size-a', '10.0',
        '--min-shift-a', '100.0',
        '--max-shift-a', '180.0',
        '--seed', '2',
        '--n-processes', '1',
    ]

# TestDecoyCommand: class containing unit tests for test_decoy_command.py
class TestDecoyCommand:
    def test_decoy_rejected_surface_end_to_end(self, tmp_path: Path) -> None:
        '''Rejected-surface runs end to end and writes a decoy set.'''
        seg_dir, raw_dir, particle_set_path = _run_pick(tmp_path)
        output_dir = tmp_path / 'decoy'

        result = runner.invoke(
            stamp_app,
            [
                '-d', str(tmp_path),
                'decoy',
                '--method', 'rejected-surface',
                '--real-particle-set', str(particle_set_path),
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', tmp_path,
                '--voxel-size-a', '10.0',
                '--picker-params', json.dumps({'n_mad': 2.5, 'offset_windows_angstrom': [[20.0, 80.0]]}),
                '--n-decoys-per-tomogram', '20',
                '--seed', '1',
            ],
        )
        assert result.exit_code == 0, result.output

        decoy_set = ParticleSet.model_validate(
            json.loads((tmp_path / 'stamp' / 'decoy' / 'decoy_particle_set.json').read_text())
        )
        assert decoy_set.particles
        assert all(p.source_picker.startswith('decoy-') for p in decoy_set.particles)

    def test_decoy_shifted_end_to_end(self, tmp_path: Path) -> None:
        '''Shifted runs end to end without error.'''
        seg_dir, raw_dir, particle_set_path = _run_pick(tmp_path)
        result = runner.invoke(
            stamp_app,
            [
                '-d', str(tmp_path),
                'decoy',
                '--method', 'shifted',
                '--real-particle-set', str(particle_set_path),
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', tmp_path,
                '--voxel-size-a', '10.0',
                '--min-shift-a', '100.0',
                '--max-shift-a', '180.0',
                '--seed', '2',
            ],
        )
        assert result.exit_code == 0, result.output

    def test_decoy_synthetic_noise_end_to_end(self, tmp_path: Path) -> None:
        '''Synthetic-noise writes the decoy set, manifests and volume dirs.'''
        output_dir = tmp_path / 'decoy_noise'
        result = runner.invoke(
            stamp_app,
            [
                '-d', str(tmp_path),
                'decoy',
                '--method', 'synthetic-noise',
                '--out-dir', tmp_path,
                '--voxel-size-a', '10.0',
                '--n-synthetic-tomograms', '2',
                '--synthetic-shape', '40,40,40',
                '--n-decoys-per-tomogram', '5',
                '--seed', '3',
            ],
        )
        output_dir = tmp_path / 'stamp' / 'decoy'
        assert result.exit_code == 0, result.output
        assert (output_dir / 'decoy_particle_set.json').exists()
        assert (output_dir / 'decoy_manifests.json').exists()
        assert (output_dir / 'segmentations').is_dir()
        assert (output_dir / 'tomograms').is_dir()

    def test_decoy_rejects_undersized_set(self, tmp_path: Path) -> None:
        '''Too few decoys per tomogram fails comparability check.'''
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        shape = (60, 60, 60)
        centre = np.array(shape) / 2.0
        grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1)
        distance = np.linalg.norm(grid - centre, axis=-1)
        segmentation = ((distance > 16.0) & (distance < 20.0)).astype(np.float32)
        tomogram = np.zeros(shape, dtype=np.float32)
        tomogram[segmentation > 0] = -1.0

        rng = np.random.default_rng(3)
        for _ in range(8):
            z, y, x = rng.integers(10, 50, 3)
            tomogram[z - 2:z + 3, y - 2:y + 3, x - 2:x + 3] = -8.0

        seg_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        with mrcfile.new(seg_dir / 'tomo000.mrc', overwrite=True) as mrc:
            mrc.set_data(segmentation)
        with mrcfile.new(raw_dir / 'tomo000.mrc', overwrite=True) as mrc:
            mrc.set_data(tomogram)

        pick_result = runner.invoke(
            stamp_app,
            [
                '-d', str(tmp_path),
                'pick',
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', tmp_path,
                '--voxel-size-a', '10.0',
                '--picker-params', json.dumps({'n_mad': 2.0, 'offset_windows_angstrom': [[20.0, 80.0]]}),
                '--backend', 'local',
                'stamp-native'
            ],
        )
        assert pick_result.exit_code == 0, pick_result.output
        particle_set_path = tmp_path / 'stamp' / 'pick' / 'particle_set.json'
        real_set = ParticleSet.model_validate(json.loads(particle_set_path.read_text()))
        assert len(real_set.particles) >= 3

        output_dir = tmp_path / 'decoy_undersized'
        result = runner.invoke(
            stamp_app,
            [
                '-d', str(tmp_path),
                'decoy',
                '--method', 'rejected-surface',
                '--real-particle-set', str(particle_set_path),
                '--seg-dir', str(seg_dir),
                '--raw-dir', str(raw_dir),
                '--out-dir', tmp_path,
                '--voxel-size-a', '10.0',
                '--picker-params', json.dumps({'n_mad': 2.0, 'offset_windows_angstrom': [[20.0, 80.0]]}),
                '--n-decoys-per-tomogram', '1',
                '--seed', '1',
            ],
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, StampPipelineError)
        assert 'less than half the size of the real set' in str(result.exception)
        assert not (tmp_path / 'stamp' / 'decoy' / 'decoy_particle_set.json').exists()

    def test_decoy_synthetic_noise_reuses_finished_volumes(self, tmp_path: Path) -> None:
        '''A rerun over a kept raw dir reuses finished volumes and gives identical decoys.'''
        args = [
            '-d', str(tmp_path),
            'decoy',
            '--method', 'synthetic-noise',
            '--out-dir', str(tmp_path),
            '--voxel-size-a', '10.0',
            '--n-synthetic-tomograms', '2',
            '--synthetic-shape', '40,40,40',
            '--n-decoys-per-tomogram', '5',
            '--seed', '3',
            '--n-processes', '1',
            '--keep-raw',
        ]
        output_dir = tmp_path / 'stamp' / 'decoy'
        assert runner.invoke(stamp_app, args).exit_code == 0
        volume = output_dir / 'tomograms' / 'decoy-noise-000.mrc'
        modified = volume.stat().st_mtime_ns
        first = _positions(output_dir / 'decoy_particle_set.json')

        assert runner.invoke(stamp_app, args).exit_code == 0
        assert volume.stat().st_mtime_ns == modified
        assert _positions(output_dir / 'decoy_particle_set.json') == first

    def test_decoy_rejected_surface_resumes_after_interruption(self, tmp_path: Path, monkeypatch) -> None:
        '''Scoring resumes from finished tomograms and the decoys match an uninterrupted run.'''
        from stamp.decoy import generate
        seg_dir, raw_dir, particle_set_path = _run_pick(tmp_path, n_tomograms=3)
        real = generate._score_surface
        scored: list[str] = []
        state = {'fail': True}

        # flaky: fail on the second tomogram until told otherwise, else score normally
        def flaky(manifest, config):
            if state['fail'] and len(scored) == 1:
                raise RuntimeError('boom')
            scored.append(manifest.tomogram_id)
            return real(manifest, config)

        monkeypatch.setattr(generate, '_score_surface', flaky)
        args = _rejected_surface_args(tmp_path, tmp_path, seg_dir, raw_dir, particle_set_path)
        assert runner.invoke(stamp_app, args).exit_code != 0
        finished = list(scored)
        assert len(finished) == 1

        state['fail'] = False
        scored.clear()
        assert runner.invoke(stamp_app, args).exit_code == 0
        assert len(scored) == 2 and not set(scored) & set(finished)

        clean = _rejected_surface_args(tmp_path, tmp_path / 'clean', seg_dir, raw_dir, particle_set_path)
        assert runner.invoke(stamp_app, clean).exit_code == 0
        assert _positions(tmp_path / 'stamp' / 'decoy' / 'decoy_particle_set.json') == _positions(tmp_path / 'clean' / 'stamp' / 'decoy' / 'decoy_particle_set.json')

    def test_decoy_shifted_resumes_after_interruption(self, tmp_path: Path, monkeypatch) -> None:
        '''Surface extraction resumes from finished tomograms and the decoys match an uninterrupted run.'''
        from stamp.decoy import generate
        seg_dir, raw_dir, particle_set_path = _run_pick(tmp_path, n_tomograms=3)
        real = generate._extract_surface_for_manifest
        extracted: list[str] = []
        state = {'fail': True}

        # flaky: fail on the second tomogram until told otherwise, else extract normally
        def flaky(manifest):
            if state['fail'] and len(extracted) == 1:
                raise RuntimeError('boom')
            extracted.append(manifest.tomogram_id)
            return real(manifest)

        monkeypatch.setattr(generate, '_extract_surface_for_manifest', flaky)
        args = _shifted_args(tmp_path, tmp_path, seg_dir, raw_dir, particle_set_path)
        assert runner.invoke(stamp_app, args).exit_code != 0
        finished = list(extracted)
        assert len(finished) == 1

        state['fail'] = False
        extracted.clear()
        assert runner.invoke(stamp_app, args).exit_code == 0
        assert len(extracted) == 2 and not set(extracted) & set(finished)

        clean = _shifted_args(tmp_path, tmp_path / 'clean', seg_dir, raw_dir, particle_set_path)
        assert runner.invoke(stamp_app, clean).exit_code == 0
        assert _positions(tmp_path / 'stamp' / 'decoy' / 'decoy_particle_set.json') == _positions(tmp_path / 'clean' / 'stamp' / 'decoy' / 'decoy_particle_set.json')
