'''
STAMP: integration tests for `stamp stream append`
'''

# Import external dependencies
import json, mrcfile, numpy as np, zlib
from pathlib import Path
from typer.testing import CliRunner

# Import internal STAMP objects
from stamp.cli.cli import stamp_app
from stamp.classify.extract import extract_subvolume
from stamp.stream.state import StreamDir
from stamp.utils.errors import StampError, StampPipelineError, StampValidationError

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
    # seeded noise per tomogram, a density blob at each particle (array index is z, y, x), so subvolumes are not constant after membrane subtraction
    tomogram = np.random.default_rng(zlib.crc32(tomogram_id.encode())).normal(0.0, 1.0, shape).astype(np.float32)
    tomogram[segmentation > 0] -= 1.0
    for x, y, z in PARTICLES_XYZ:
        tomogram[z - 2:z + 3, y - 2:y + 3, x - 2:x + 3] -= 6.0
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

    def test_plots_written_on_append_then_tomogram_deleted(self, tmp_path: Path) -> None:
        '''With --plots, the tomogram's plots exist after append, and the tomogram can then be deleted.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0'])
        seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't0')
        result = runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw, '--plots', '--no-pick-zstack-movie'))
        assert result.exit_code == 0, result.output
        tomogram_dir = _stream(tmp_path).tomogram_dir('t0')
        assert any((tomogram_dir / 'plots').glob('*'))
        assert any((tomogram_dir / 'decoy' / 'plots').glob('*'))
        seg.unlink()
        raw.unlink()
        assert _stream(tmp_path).is_done('t0')

    def test_synthetic_noise_is_rejected_with_pointer(self, tmp_path: Path) -> None:
        '''--decoy-method synthetic-noise is refused, pointing at stamp decoy.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0'])
        seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't0')
        result = runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw, '--decoy-method', 'synthetic-noise'))
        assert isinstance(result.exception, StampError) and 'stamp decoy' in str(result.exception)


# _rows: comparable (id, tomogram, position, orientation, prior, picker, half-set) rows from a particle set JSON, sorted by id
def _rows(path: Path) -> list[list]:
    keys = ['particle_id', 'tomogram_id', 'position', 'orientation', 'prior', 'source_picker', 'half_set']
    return sorted([[particle[key] for key in keys] for particle in json.loads(path.read_text())['particles']], key=lambda row: row[0])

# _dataset: three tomograms with a dataset-wide picks file, returning (picks, seg dir, raw dir)
def _dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    ids = ['t0', 't1', 't2']
    seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
    for tomogram_id in ids:
        _write_pair(seg_dir, raw_dir, tomogram_id)
    return _write_picks(tmp_path / 'picks.csv', ids), seg_dir, raw_dir

# TestStreamFinalise: class checking that finalise merges the appended tomograms into the same sets as pick and decoy
class TestStreamFinalise:
    def test_finalised_sets_match_pick_and_decoy(self, tmp_path: Path) -> None:
        '''Finalised sets equal the sets stamp pick and stamp decoy produce over the same inputs.'''
        picks, seg_dir, raw_dir = _dataset(tmp_path)
        pick_out, decoy_out = tmp_path / 'pick_out', tmp_path / 'decoy_out'
        result = runner.invoke(stamp_app, ['-d', str(tmp_path), 'pick', str(picks), '--seg-dir', str(seg_dir), '--raw-dir', str(raw_dir), '--out-dir', str(pick_out), '--voxel-size-a', '10.0'])
        assert result.exit_code == 0, result.output
        pick_set = pick_out / 'stamp' / 'pick' / 'particle_set.json'
        result = runner.invoke(stamp_app, ['-d', str(tmp_path), 'decoy', '--method', 'auto', '--real-particle-set', str(pick_set), '--seg-dir', str(seg_dir), '--raw-dir', str(raw_dir), '--out-dir', str(decoy_out), '--voxel-size-a', '10.0'])
        assert result.exit_code == 0, result.output

        for tomogram_id in ['t0', 't1', 't2']:
            result = runner.invoke(stamp_app, _append_args(tmp_path, picks, seg_dir / f'{tomogram_id}.mrc', raw_dir / f'{tomogram_id}.mrc'))
            assert result.exit_code == 0, result.output
        result = runner.invoke(stamp_app, ['-d', str(tmp_path), 'stream', 'finalise', '--out-dir', str(tmp_path)])
        assert result.exit_code == 0, result.output

        stream = tmp_path / 'stamp' / 'stream'
        assert _rows(stream / 'particle_set.json') == _rows(pick_set)
        assert _rows(stream / 'decoy_particle_set.json') == _rows(decoy_out / 'stamp' / 'decoy' / 'decoy_particle_set.json')

    def test_whole_tomograms_stay_in_one_half(self, tmp_path: Path) -> None:
        '''Every tomogram's particles share one half-set, and the halves are as even as whole tomograms allow.'''
        picks, seg_dir, raw_dir = _dataset(tmp_path)
        for tomogram_id in ['t0', 't1', 't2']:
            assert runner.invoke(stamp_app, _append_args(tmp_path, picks, seg_dir / f'{tomogram_id}.mrc', raw_dir / f'{tomogram_id}.mrc')).exit_code == 0
        assert runner.invoke(stamp_app, ['-d', str(tmp_path), 'stream', 'finalise', '--out-dir', str(tmp_path)]).exit_code == 0
        rows = _rows(tmp_path / 'stamp' / 'stream' / 'particle_set.json')
        halves_by_tomogram: dict[str, set[str]] = {}
        for row in rows:
            halves_by_tomogram.setdefault(row[1], set()).add(row[6])
        assert all(len(halves) == 1 for halves in halves_by_tomogram.values())
        counts = [sum(row[6] == half for row in rows) for half in ('A', 'B')]
        assert abs(counts[0] - counts[1]) <= len(PARTICLES_XYZ)

    def test_comparability_failure_raises_with_decoy_count(self, tmp_path: Path) -> None:
        '''A decoy set under half the real size is refused with the count message.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0'])
        seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't0')
        assert runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw)).exit_code == 0
        decoy_path = _stream(tmp_path).tomogram_dir('t0') / 'decoy_particles.json'
        payload = json.loads(decoy_path.read_text())
        payload['particles'] = payload['particles'][:1]
        decoy_path.write_text(json.dumps(payload))
        result = runner.invoke(stamp_app, ['-d', str(tmp_path), 'stream', 'finalise', '--out-dir', str(tmp_path)])
        assert isinstance(result.exception, StampError) and 'less than half' in str(result.exception)
        assert not _stream(tmp_path).settings_path.read_text().count('"finalised": true')

    def test_append_after_finalise_is_rejected(self, tmp_path: Path) -> None:
        '''Once finalised, a stream refuses further appends.'''
        picks, seg_dir, raw_dir = _dataset(tmp_path)
        assert runner.invoke(stamp_app, _append_args(tmp_path, picks, seg_dir / 't0.mrc', raw_dir / 't0.mrc')).exit_code == 0
        assert runner.invoke(stamp_app, ['-d', str(tmp_path), 'stream', 'finalise', '--out-dir', str(tmp_path)]).exit_code == 0
        result = runner.invoke(stamp_app, _append_args(tmp_path, picks, seg_dir / 't1.mrc', raw_dir / 't1.mrc'))
        assert isinstance(result.exception, StampValidationError) and 'finalised' in str(result.exception)

# _partition: particle ids grouped by class, as a set of frozensets so cluster label permutation does not matter
def _partition(path: Path) -> set[frozenset[str]]:
    groups: dict[str, set[str]] = {}
    for row in json.loads(path.read_text()):
        groups.setdefault(row['cluster_id'], set()).add(row['particle_id'])
    return {frozenset(members) for members in groups.values()}

# _finalised_stream: append all three tomograms, deleting each tomogram and segmentation afterwards, then finalise
def _finalised_stream(tmp_path: Path, *, delete: bool = True) -> None:
    picks, seg_dir, raw_dir = _dataset(tmp_path)
    # decoy geometry scaled to the 60-voxel test volume, so shifted decoys fit inside it and can be extracted
    small_decoys = ['--min-shift-a', '30', '--max-shift-a', '60', '--min-distance-from-real-a', '20', '--min-distance-from-picks-a', '25']
    for tomogram_id in ['t0', 't1', 't2']:
        seg, raw = seg_dir / f'{tomogram_id}.mrc', raw_dir / f'{tomogram_id}.mrc'
        assert runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw, *small_decoys)).exit_code == 0
        if delete:
            seg.unlink()
            raw.unlink()
    assert runner.invoke(stamp_app, ['-d', str(tmp_path), 'stream', 'finalise', '--out-dir', str(tmp_path)]).exit_code == 0

# CLASSIFY_ARGS: deterministic, small-data classify settings shared by stamp classify and stream classify (stream classify takes its box from stream.json)
CLASSIFY_ARGS = ['--method', 'kmeans', '--n-clusters', '2', '--min-cluster-size', '2', '--seed', '0', '--no-plots']

# TestStreamClassify: class checking that stream classify runs from the cache and matches stamp classify
class TestStreamClassify:
    def test_unfinalised_stream_is_rejected(self, tmp_path: Path) -> None:
        '''stream classify refuses a stream that has not been finalised.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0'])
        seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't0')
        assert runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw)).exit_code == 0
        result = runner.invoke(stamp_app, ['-d', str(tmp_path), 'stream', 'classify', '--out-dir', str(tmp_path), *CLASSIFY_ARGS])
        assert isinstance(result.exception, StampError) and 'finalise' in str(result.exception)

    def test_classify_runs_with_all_tomograms_deleted(self, tmp_path: Path) -> None:
        '''Tomograms and segmentations are deleted at append time; stream classify still succeeds from the cache.'''
        _finalised_stream(tmp_path)
        result = runner.invoke(stamp_app, ['-d', str(tmp_path), 'stream', 'classify', '--out-dir', str(tmp_path), *CLASSIFY_ARGS])
        assert result.exit_code == 0, result.output
        assert (tmp_path / 'stamp' / 'stream' / 'classify' / 'real' / 'class_assignments.json').is_file()
        assert (tmp_path / 'stamp' / 'stream' / 'classify' / 'decoy' / 'class_assignments.json').is_file()

    def test_missing_marker_raises(self, tmp_path: Path) -> None:
        '''A missing subvolume marker is a hard error, since the tomogram is gone.'''
        _finalised_stream(tmp_path)
        (tmp_path / 'stamp' / 'stream' / 'subvolumes' / 'real' / 'normal' / 't1.json').unlink()
        result = runner.invoke(stamp_app, ['-d', str(tmp_path), 'stream', 'classify', '--out-dir', str(tmp_path), *CLASSIFY_ARGS])
        assert isinstance(result.exception, StampPipelineError) and 't1' in str(result.exception)

    def test_matches_stamp_classify_on_the_same_inputs(self, tmp_path: Path) -> None:
        '''Stream classify partitions the real particles exactly as stamp classify does on the finalised set.'''
        _finalised_stream(tmp_path, delete=False)
        stream_dir = tmp_path / 'stamp' / 'stream'
        result = runner.invoke(stamp_app, ['-d', str(tmp_path), 'stream', 'classify', '--out-dir', str(tmp_path), *CLASSIFY_ARGS])
        assert result.exit_code == 0, result.output
        classify_out = tmp_path / 'classify_out'
        result = runner.invoke(stamp_app, ['-d', str(tmp_path), 'classify', '-p', str(stream_dir / 'particle_set.json'), '-r', str(tmp_path / 'raw'), '-s', str(tmp_path / 'seg'), '-o', str(classify_out), '--voxel-size-a', '10.0', '--box-length-a', '150', *CLASSIFY_ARGS])
        assert result.exit_code == 0, result.output
        assert _partition(stream_dir / 'classify' / 'real' / 'class_assignments.json') == _partition(classify_out / 'stamp' / 'classify' / 'class_assignments.json')

# TestStreamExport: class checking the optional per-particle raw box export on append
class TestStreamExport:
    def test_export_writes_one_box_per_kept_real_particle(self, tmp_path: Path) -> None:
        '''One MRC per kept real particle, at the stream box and voxel size; the scratch cache is removed.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0'])
        seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't0')
        result = runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw, '--export-particles'))
        assert result.exit_code == 0, result.output
        stream = _stream(tmp_path)
        done = json.loads((stream.tomogram_dir('t0') / 'done.json').read_text())
        files = sorted(stream.particle_dir('t0').glob('*.mrc'))
        assert len(files) == sum(done['n_kept_by_prior']['real'].values()) > 0
        with mrcfile.open(files[0]) as mrc:
            assert mrc.data.shape == (15, 15, 15)
            assert abs(float(mrc.voxel_size.x) - 10.0) < 1e-3
        assert not (stream.particle_dir('t0') / '.scratch').exists()

    def test_exported_box_matches_raw_extraction(self, tmp_path: Path) -> None:
        '''Each exported box equals extract_subvolume of the raw tomogram at the particle's position and orientation.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0'])
        seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't0')
        raw_data = np.array(mrcfile.open(raw).data, dtype=np.float32)
        result = runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw, '--export-particles'))
        assert result.exit_code == 0, result.output
        stream = _stream(tmp_path)
        particles = {p['particle_id']: p for p in json.loads((stream.tomogram_dir('t0') / 'particles.json').read_text())['particles']}
        files = sorted(stream.particle_dir('t0').glob('*.mrc'))
        assert files
        for path in files:
            particle = particles[path.stem]
            orientation = tuple(particle['orientation']) if particle['orientation'] is not None else None
            expected = extract_subvolume(raw_data, tuple(particle['position']), orientation, 15)
            with mrcfile.open(path) as mrc:
                np.testing.assert_allclose(mrc.data, expected, rtol=1e-5, atol=1e-5)

    def test_export_off_writes_no_particles_directory(self, tmp_path: Path) -> None:
        '''Without --export-particles there is no particles/ directory.'''
        picks = _write_picks(tmp_path / 'picks.csv', ['t0'])
        seg, raw = _write_pair(tmp_path / 'seg', tmp_path / 'raw', 't0')
        assert runner.invoke(stamp_app, _append_args(tmp_path, picks, seg, raw)).exit_code == 0
        assert not (_stream(tmp_path).root / 'particles').exists()

# DECOY_SMALL_STREAM / DECOY_SMALL_CLI: the same scaled-down decoy geometry for both paths (stream and decoy spell the picks separation flag differently)
DECOY_SMALL_STREAM = ['--min-shift-a', '30', '--max-shift-a', '60', '--min-distance-from-real-a', '20', '--min-distance-from-picks-a', '25']
DECOY_SMALL_CLI = ['--min-shift-a', '30', '--max-shift-a', '60', '--min-distance-from-real-a', '20', '--min-pick-distance-a', '25']

# _write_candidates: two tiny structures and a candidates YAML for identify
def _write_candidates(root: Path) -> Path:
    (root / 'small.pdb').write_text('ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n')
    (root / 'large.pdb').write_text('\n'.join(f'ATOM  {i:>5}  CA  ALA A{i:>4}    {i * 2.0:8.3f}   0.000   0.000  1.00  0.00           C' for i in range(1, 40)) + '\n')
    path = root / 'candidates.yaml'
    path.write_text('candidates:\n  - name: small\n    structure_path: small.pdb\n  - name: large\n    structure_path: large.pdb\n')
    return path

# TestStreamEquivalence: class checking that the stream path and the normal path agree on sets, classes and identification over the same data
class TestStreamEquivalence:
    def test_stream_and_normal_paths_agree(self, tmp_path: Path) -> None:
        '''Normal pick, decoy, classify and identify agree with the appended, finalised and streamed-classified path.'''
        picks, seg_dir, raw_dir = _dataset(tmp_path)
        candidates = _write_candidates(tmp_path)
        normal, streamed = tmp_path / 'normal', tmp_path / 'streamed'
        ids = ['t0', 't1', 't2']

        # normal path, while every tomogram is still on disk (classify reads them)
        result = runner.invoke(stamp_app, ['-d', str(normal), 'pick', str(picks), '--seg-dir', str(seg_dir), '--raw-dir', str(raw_dir), '--out-dir', str(normal), '--voxel-size-a', '10.0'])
        assert result.exit_code == 0, result.output
        pick_set = normal / 'stamp' / 'pick' / 'particle_set.json'
        result = runner.invoke(stamp_app, ['-d', str(normal), 'decoy', '--method', 'auto', '--real-particle-set', str(pick_set), '--seg-dir', str(seg_dir), '--raw-dir', str(raw_dir), '--out-dir', str(normal), '--voxel-size-a', '10.0', *DECOY_SMALL_CLI])
        assert result.exit_code == 0, result.output
        decoy_set = normal / 'stamp' / 'decoy' / 'decoy_particle_set.json'
        for particles in (pick_set, decoy_set):
            result = runner.invoke(stamp_app, ['-d', str(normal), 'classify', '-p', str(particles), '-r', str(raw_dir), '-s', str(seg_dir), '-o', str(normal), '--voxel-size-a', '10.0', '--box-length-a', '150', *CLASSIFY_ARGS])
            assert result.exit_code == 0, result.output

        # stream path, deleting each tomogram after its append
        picks_stream_root = streamed
        for tomogram_id in ids:
            seg, raw = seg_dir / f'{tomogram_id}.mrc', raw_dir / f'{tomogram_id}.mrc'
            assert runner.invoke(stamp_app, _append_args(picks_stream_root, picks, seg, raw, *DECOY_SMALL_STREAM)).exit_code == 0
            seg.unlink()
            raw.unlink()
        assert runner.invoke(stamp_app, ['-d', str(streamed), 'stream', 'finalise', '--out-dir', str(streamed)]).exit_code == 0
        assert runner.invoke(stamp_app, ['-d', str(streamed), 'stream', 'classify', '--out-dir', str(streamed), *CLASSIFY_ARGS]).exit_code == 0
        stream = streamed / 'stamp' / 'stream'

        # particle sets: ids, positions, orientations, priors, pickers and half-sets
        assert _rows(stream / 'particle_set.json') == _rows(pick_set)
        assert _rows(stream / 'decoy_particle_set.json') == _rows(decoy_set)
        assert json.loads((stream / 'particle_set.json').read_text())['contributing_pickers'] == json.loads(pick_set.read_text())['contributing_pickers']
        assert json.loads((stream / 'decoy_particle_set.json').read_text())['contributing_pickers'] == json.loads(decoy_set.read_text())['contributing_pickers']

        # classes: same partitions for real and decoy
        assert _partition(stream / 'classify' / 'real' / 'class_assignments.json') == _partition(normal / 'stamp' / 'classify' / 'class_assignments.json')
        assert _partition(stream / 'classify' / 'decoy' / 'class_assignments.json') == _partition(normal / 'stamp' / 'classify_decoy' / 'class_assignments.json')

        # identification: same identified classes and candidates on both paths
        for root, classes, decoys in (
            (normal, normal / 'stamp' / 'classify' / 'class_averages', normal / 'stamp' / 'classify_decoy' / 'class_averages'),
            (streamed, stream / 'classify' / 'real' / 'class_averages', stream / 'classify' / 'decoy' / 'class_averages'),
        ):
            result = runner.invoke(stamp_app, ['-d', str(root), 'identify', '--classes', str(classes), '--decoy-classes', str(decoys), '--candidates', str(candidates), '--out-dir', str(root / 'identify'), '--resolution', '30.0', '-n', '1', '--no-plots'])
            assert result.exit_code == 0, result.output
        identified = lambda root: {(row['cluster_id'], row['candidate_protein']) for row in json.loads((root / 'identify' / 'stamp' / 'identify' / 'identification.json').read_text())}
        assert identified(normal) == identified(streamed) and identified(normal)
