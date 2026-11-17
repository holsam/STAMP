'''
STAMP: integration tests for `stamp stream append`
'''

# Import external dependencies
import json, mrcfile, numpy as np
from pathlib import Path
from typer.testing import CliRunner

# Import internal STAMP objects
from stamp.cli.cli import stamp_app
from stamp.stream.state import StreamDir
from stamp.utils.errors import StampError, StampValidationError

# Initialise runner
runner = CliRunner()

# PARTICLES_XYZ: six particle positions (x, y, z) inside the 60^3 vesicle, 14 voxels from its centre
PARTICLES_XYZ = ((30, 30, 44), (30, 30, 16), (30, 44, 30), (30, 16, 30), (44, 30, 30), (16, 30, 30))

# _write_pair: write a vesicle as a matched segmentation/tomogram pair at 10 Å per voxel
def _write_pair(seg_dir: Path, raw_dir: Path, tomogram_id: str) -> tuple[Path, Path]:
    shape = (60, 60, 60)
    centre = np.array(shape) / 2.0
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1)
    distance = np.linalg.norm(grid - centre, axis=-1)
    segmentation = ((distance > 16.0) & (distance < 20.0)).astype(np.float32)
    tomogram = np.zeros(shape, dtype=np.float32)
    tomogram[segmentation > 0] = -1.0
    seg_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    seg_path, raw_path = seg_dir / f'{tomogram_id}.mrc', raw_dir / f'{tomogram_id}.mrc'
    for path, data in ((seg_path, segmentation), (raw_path, tomogram)):
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = 10.0
    return seg_path, raw_path

# _write_picks: one dataset-wide CSV with PARTICLES_XYZ in each given tomogram
def _write_picks(path: Path, tomogram_ids: list[str]) -> Path:
    rows = ['tomogram_id,x,y,z'] + [f'{tomogram_id},{x},{y},{z}' for tomogram_id in tomogram_ids for x, y, z in PARTICLES_XYZ]
    path.write_text('\n'.join(rows) + '\n')
    return path

# _append_args: CLI arguments for one append call; box 150 Å at 10 Å/voxel is a 15-voxel box
def _append_args(tmp_path: Path, picks: Path, seg: Path, raw: Path, *extra: str) -> list[str]:
    return [
        '-d', str(tmp_path),
        'stream', 'append', str(picks),
        '-r', str(raw), '-s', str(seg),
        '--out-dir', str(tmp_path),
        '--voxel-size-a', '10.0', '--box-length-a', '150',
        *extra,
    ]

# _stream: the stream directory an append writes into under the out dir
def _stream(tmp_path: Path) -> StreamDir:
    return StreamDir(tmp_path / 'stamp' / 'stream')

# TestStreamAppend: class containing integration tests for stream append
class TestStreamAppend:
    def test_append_each_tomogram_then_delete_it(self, tmp_path: Path) -> None:
        '''Each tomogram is appended, then its raw and segmentation are deleted; the stream keeps everything it needs.'''
        ids = ['t0', 't1', 't2']
        picks = _write_picks(tmp_path / 'picks.csv', ids)
        for tomogram_id in ids:
            seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', tomogram_id)
            result = runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw))
            assert result.exit_code == 0, result.output
            seg.unlink()
            raw.unlink()
        stream = _stream(tmp_path)
        assert stream.done_tomograms() == ids
        assert any((stream.root / 'subvolumes' / 'real' / 'normal').glob('*'))

    def test_dataset_wide_picks_give_only_this_tomograms_particles(self, tmp_path: Path) -> None:
        '''A picks file covering two tomograms yields particles for only the appended one, with per-tomogram ids.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0', 't1'])
        seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't1')
        result = runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw))
        assert result.exit_code == 0, result.output
        particles = json.loads((_stream(tmp_path).tomogram_dir('t1') / 'particles.json').read_text())['particles']
        assert len(particles) == len(PARTICLES_XYZ)
        assert {p['tomogram_id'] for p in particles} == {'t1'}
        assert particles[0]['particle_id'] == 't1_p0000'

    def test_reappending_is_a_no_op(self, tmp_path: Path) -> None:
        '''A tomogram already marked done is skipped and its outputs are left as they were.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0'])
        seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't0')
        assert runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw)).exit_code == 0
        done = _stream(tmp_path).tomogram_dir('t0') / 'done.json'
        before = done.read_text()
        assert runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw)).exit_code == 0
        assert done.read_text() == before

    def test_changed_box_is_rejected(self, tmp_path: Path) -> None:
        '''A later append with a different box is refused, naming the box setting.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0', 't1'])
        seg0, raw0 = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't0')
        seg1, raw1 = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't1')
        assert runner.invoke(stamp_app, _append_args(tmp_path, picks, seg0, raw0)).exit_code == 0
        result = runner.invoke(stamp_app, _append_args(tmp_path, picks, seg1, raw1, '--box-length-a', '250'))
        assert isinstance(result.exception, StampValidationError) and 'box_voxels' in str(result.exception)

    def test_zero_particle_tomogram_is_recorded_empty(self, tmp_path: Path) -> None:
        '''A tomogram with no picks warns, writes an empty particles.json and is marked done.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0'])
        seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 'empty')
        result = runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw))
        assert result.exit_code == 0, result.output
        stream = _stream(tmp_path)
        assert stream.is_done('empty')
        assert json.loads((stream.tomogram_dir('empty') / 'particles.json').read_text())['particles'] == []

    def test_synthetic_noise_is_rejected_with_pointer(self, tmp_path: Path) -> None:
        '''--decoy-method synthetic-noise is refused, pointing at stamp decoy.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0'])
        seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't0')
        result = runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw, '--decoy-method', 'synthetic-noise'))
        assert isinstance(result.exception, StampError) and 'stamp decoy' in str(result.exception)
