'''
STAMP: integration tests for `stamp refine prepare` and `stamp refine assess`
'''

# Import external dependencies
import json, mrcfile, numpy as np, pandas as pd, pytest, starfile
from typer.testing import CliRunner
from stamp.cli.cli import stamp_app
from stamp.utils.errors import StampValidationError

# Initialise runner
runner = CliRunner()

# _fixture: a two-half class with per-half Stage D averages, plus the three input JSON files
def _fixture(tmp_path, mixed=False):
    stage_d = tmp_path / 'stage_d'
    averages = stage_d / 'class_averages'
    averages.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for half in ('A', 'B'):
        with mrcfile.new(averages / f'c00_half{half}_n20.mrc', overwrite=True) as mrc:
            mrc.set_data(rng.random((16, 16, 16)).astype(np.float32))
            mrc.voxel_size = 3.0

    halves = ['A', 'A'] if mixed else ['A', 'B']
    particles = {
        'particles': [
            {
                'particle_id': f'p{i}',
                'tomogram_id': 't0',
                'position': [8.0, 8.0, 8.0],
                'source_picker': 'x',
                'half_set': halves[i],
                'prior': 'normal',
            }
            for i in range(2)
        ],
        'consensus_rule': 'intersection', 
        'contributing_pickers': ['x'],
    }
    (tmp_path / 'particle_set.json').write_text(json.dumps(particles))
    (stage_d / 'class_assignments.json').write_text(json.dumps(
        [
            {
                'particle_id': f'p{i}',
                'cluster_id': 'c00',
                'classifier': 'k'
            } 
        for i in range(2)]
    ))
    (tmp_path / 'identification.json').write_text(json.dumps(
        [
            {
                'cluster_id': 'c00',
                'candidate_protein': 'x',
                'fit_score': 0.9,
                'method': 'm',
                'score_gap_to_runner_up': 0.3
            }
        ]
    ))
    (tmp_path / 'tomo').mkdir()
    # raw tomogram header gives the size for centred coordinates: nx=40, ny=30, nz=20
    with mrcfile.new(tmp_path / 'tomo' / 't0.mrc', overwrite=True) as mrc:
        mrc.set_data(np.zeros((20, 30, 40), dtype=np.float32))
        mrc.voxel_size = 3.0
    return stage_d

def _prepare(tmp_path, stage_d, *extra):
    return runner.invoke(stamp_app, [
        '-d', str(tmp_path),
        'refine', 'prepare',
        '--class-id', 'c00',
        '--identification', str(tmp_path / 'identification.json'),
        '--particles', str(tmp_path / 'particle_set.json'),
        '--class-assignments', str(stage_d / 'class_assignments.json'),
        '--raw-dir', str(tmp_path / 'tomo'),
        '--out-dir', str(tmp_path),
        '--voxel-size-a', '3.0',
        *extra,
    ])

def _assess(tmp_path, half_a, half_b, *extra):
    return runner.invoke(stamp_app, [
        '-d', str(tmp_path),
        'refine', 'assess',
        '--class-id', 'c00',
        '--prepared-dir', str(tmp_path / 'stamp' / 'refine' / 'c00'),
        '--half-a', str(half_a),
        '--half-b', str(half_b),
        *extra,
    ])

def _write_refined(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = 3.0

# TestRefinePrepare: class containing tests for `stamp refine prepare`
class TestRefinePrepare:
    def test_prepare_writes_disjoint_halves(self, tmp_path):
        stage_d = _fixture(tmp_path)
        result = _prepare(tmp_path, stage_d)
        assert result.exit_code == 0, result.output
        combined = tmp_path / 'stamp' / 'refine' / 'c00'
        for half in ('halfA', 'halfB'):
            assert (combined / half / 'particles.star').is_file()
            assert (combined / half / 'reference.mrc').is_file()
            assert (combined / half / 'README.md').is_file()
        assert json.loads((combined / 'prepare.json').read_text())['voxel_size_angstrom'] == 3.0
        rows_a = starfile.read(combined / 'halfA' / 'particles.star')
        rows_b = starfile.read(combined / 'halfB' / 'particles.star')
        assert list(rows_a['rlnTomoParticleName']) == ['p0'] and list(rows_b['rlnTomoParticleName']) == ['p1']

    def test_prepare_pixel_coordinates_flag_writes_voxel_positions(self, tmp_path):
        stage_d = _fixture(tmp_path)
        assert _prepare(tmp_path, stage_d, '--coordinate-system', 'pixel').exit_code == 0
        rows = starfile.read(tmp_path / 'stamp' / 'refine' / 'c00' / 'halfA' / 'particles.star')
        assert not [c for c in rows.columns if c.startswith('rlnCenteredCoordinate')]
        assert list(rows[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].iloc[0]) == [8.0, 8.0, 8.0]

    def test_prepare_writes_centred_angstrom_coordinates_from_raw_header(self, tmp_path):
        stage_d = _fixture(tmp_path)
        assert _prepare(tmp_path, stage_d).exit_code == 0
        rows = starfile.read(tmp_path / 'stamp' / 'refine' / 'c00' / 'halfA' / 'particles.star')
        assert not [c for c in rows.columns if c.startswith('rlnCoordinate')]
        # position (8, 8, 8) in a 40 x 30 x 20 tomogram at 3 A per voxel
        expected = [(8.0 - 20.0) * 3.0, (8.0 - 15.0) * 3.0, (8.0 - 10.0) * 3.0]
        actual = [rows.iloc[0][c] for c in ('rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst', 'rlnCenteredCoordinateZAngst')]
        assert np.allclose(actual, expected)

    def test_prepare_rejects_half_set_mixed_class(self, tmp_path):
        stage_d = _fixture(tmp_path, mixed=True)
        result = _prepare(tmp_path, stage_d)
        assert result.exit_code != 0
        assert not (tmp_path / 'stamp' / 'refine' / 'c00' / 'halfA' / 'particles.star').exists()

# TestRefineAssess: class containing tests for `stamp refine assess`
class TestRefineAssess:
    def _prepared(self, tmp_path):
        stage_d = _fixture(tmp_path)
        assert _prepare(tmp_path, stage_d).exit_code == 0
        return stage_d, tmp_path / 'stamp' / 'refine' / 'c00'

    def test_assess_end_to_end(self, tmp_path):
        _, prepared = self._prepared(tmp_path)
        rng = np.random.default_rng(1)
        signal = np.asarray(mrcfile.open(prepared / 'halfA' / 'reference.mrc').data, dtype=np.float32)
        _write_refined(tmp_path / 'runA' / 'final.mrc', signal + 0.1 * rng.random(signal.shape))
        _write_refined(tmp_path / 'runB' / 'final.mrc', signal + 0.1 * rng.random(signal.shape))
        result = _assess(tmp_path, tmp_path / 'runA' / 'final.mrc', tmp_path / 'runB' / 'final.mrc')
        assert result.exit_code == 0, result.output
        assert (prepared / 'fsc.txt').is_file() and (prepared / 'fsc.svg').is_file()
        assert (prepared / 'params.toml').is_file()

    def test_assess_rejects_same_directory(self, tmp_path):
        self._prepared(tmp_path)
        rng = np.random.default_rng(2)
        _write_refined(tmp_path / 'runA' / 'final.mrc', rng.random((16, 16, 16)))
        _write_refined(tmp_path / 'runA' / 'final_b.mrc', rng.random((16, 16, 16)))
        result = _assess(tmp_path, tmp_path / 'runA' / 'final.mrc', tmp_path / 'runA' / 'final_b.mrc')
        assert result.exit_code != 0
        assert isinstance(result.exception, StampValidationError) and 'share a directory' in str(result.exception)

    def test_assess_rejects_unrefined_seed(self, tmp_path):
        _, prepared = self._prepared(tmp_path)
        rng = np.random.default_rng(3)
        seed = np.asarray(mrcfile.open(prepared / 'halfA' / 'reference.mrc').data, dtype=np.float32)
        _write_refined(tmp_path / 'runA' / 'final.mrc', seed)
        _write_refined(tmp_path / 'runB' / 'final.mrc', rng.random(seed.shape))
        result = _assess(tmp_path, tmp_path / 'runA' / 'final.mrc', tmp_path / 'runB' / 'final.mrc')
        assert result.exit_code != 0
        assert isinstance(result.exception, StampValidationError) and 'identical to its seed' in str(result.exception)

# STREAM_PARTICLES: six particle positions (x, y, z) inside the 60^3 test vesicle, 14 voxels from its centre
STREAM_PARTICLES = ((30, 30, 44), (30, 30, 16), (30, 44, 30), (30, 16, 30), (44, 30, 30), (16, 30, 30))

# _write_stream_pair: a vesicle segmentation and a noisy tomogram with a blob at each particle, deterministic per id
def _write_stream_pair(seg_dir, raw_dir, tomogram_id):
    import zlib
    shape = (60, 60, 60)
    centre = np.array(shape) / 2.0
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), axis=-1)
    segmentation = ((np.linalg.norm(grid - centre, axis=-1) > 16.0) & (np.linalg.norm(grid - centre, axis=-1) < 20.0)).astype(np.float32)
    tomogram = np.random.default_rng(zlib.crc32(tomogram_id.encode())).normal(0.0, 1.0, shape).astype(np.float32)
    tomogram[segmentation > 0] -= 1.0
    for x, y, z in STREAM_PARTICLES:
        tomogram[z - 2:z + 3, y - 2:y + 3, x - 2:x + 3] -= 6.0
    seg_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for path, data in ((seg_dir / f'{tomogram_id}.mrc', segmentation), (raw_dir / f'{tomogram_id}.mrc', tomogram)):
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = 10.0

# TestRefinePrepareFromStream: class checking that refine prepare takes tomogram sizes from a stream directory, with no tomograms on disk
class TestRefinePrepareFromStream:
    def test_stream_source_matches_raw_source_and_needs_no_tomograms(self, tmp_path):
        import shutil
        from typer.testing import CliRunner
        from stamp.cli.cli import stamp_app
        from stamp.utils.errors import StampValidationError
        cli = CliRunner()
        ids = ['t0', 't1', 't2']
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        for tomogram_id in ids:
            _write_stream_pair(seg_dir, raw_dir, tomogram_id)
        picks = tmp_path / 'picks.csv'
        picks.write_text('\n'.join(['tomogram_id,x,y,z'] + [f'{t},{x},{y},{z}' for t in ids for x, y, z in STREAM_PARTICLES]) + '\n')
        stream = tmp_path / 'stamp' / 'stream'
        small_decoys = ['--min-shift-a', '30', '--max-shift-a', '60', '--min-distance-from-real-a', '20', '--min-distance-from-picks-a', '25']
        for tomogram_id in ids:
            args = ['-d', str(tmp_path), 'stream', 'append', str(picks), '-r', str(raw_dir / f'{tomogram_id}.mrc'), '-s', str(seg_dir / f'{tomogram_id}.mrc'), '--out-dir', str(tmp_path), '--voxel-size-a', '10.0', '--box-length-a', '150', *small_decoys]
            assert cli.invoke(stamp_app, args).exit_code == 0
        assert cli.invoke(stamp_app, ['-d', str(tmp_path), 'stream', 'finalise', '--out-dir', str(tmp_path)]).exit_code == 0
        assert cli.invoke(stamp_app, ['-d', str(tmp_path), 'stream', 'classify', '--out-dir', str(tmp_path), '--method', 'kmeans', '--n-clusters', '2', '--min-cluster-size', '2', '--seed', '0', '--no-plots', '--no-strict-halfset-independence']).exit_code == 0
        assignments = stream / 'classify' / 'real' / 'class_assignments.json'
        cluster = sorted({row['cluster_id'] for row in json.loads(assignments.read_text())})[0]
        identification = tmp_path / 'identification.json'
        identification.write_text(json.dumps([{'cluster_id': cluster, 'candidate_protein': 'x', 'fit_score': 0.9}]))

        def prepare(out, *source):
            return cli.invoke(stamp_app, ['-d', str(tmp_path), 'refine', 'prepare', '--class-id', cluster, '--identification', str(identification), '--particles', str(stream / 'particle_set.json'), '--class-assignments', str(assignments), '--out-dir', str(tmp_path / out), *source])

        # raw source first, while the tomograms are still on disk
        result = prepare('out_raw', '--raw-dir', str(raw_dir), '--voxel-size-a', '10.0')
        assert result.exit_code == 0, result.output
        # now delete every tomogram and segmentation; the stream alone must be enough
        shutil.rmtree(raw_dir)
        shutil.rmtree(seg_dir)
        result = prepare('out_stream', '--stream-dir', str(stream))
        assert result.exit_code == 0, result.output
        for half in ('halfA', 'halfB'):
            raw_rows = starfile.read(tmp_path / 'out_raw' / 'stamp' / 'refine' / cluster / half / 'particles.star')
            stream_rows = starfile.read(tmp_path / 'out_stream' / 'stamp' / 'refine' / cluster / half / 'particles.star')
            pd.testing.assert_frame_equal(raw_rows, stream_rows)
        assert json.loads((tmp_path / 'out_stream' / 'stamp' / 'refine' / cluster / 'prepare.json').read_text())['voxel_size_angstrom'] == 10.0

    def test_source_must_be_exactly_one(self, tmp_path):
        from typer.testing import CliRunner
        from stamp.cli.cli import stamp_app
        cli = CliRunner()
        for name in ('x.json',):
            (tmp_path / name).write_text('[]')
        common = ['-d', str(tmp_path), 'refine', 'prepare', '--class-id', 'c00', '--identification', str(tmp_path / 'x.json'), '--particles', str(tmp_path / 'x.json'), '--class-assignments', str(tmp_path / 'x.json'), '--out-dir', str(tmp_path)]
        for source in ([], ['--raw-dir', str(tmp_path), '--stream-dir', str(tmp_path)]):
            result = cli.invoke(stamp_app, [*common, *source])
            assert isinstance(result.exception, StampValidationError) and 'exactly one' in str(result.exception)

