'''
STAMP: integration tests for `stamp pick` command
'''

# Import external dependencies
import json, mrcfile, numpy as np
from pathlib import Path
from typer.testing import CliRunner

# Import internal STAMP objects
from stamp.cli.cli import stamp_app
from stamp.schemas.particles import ParticleSet

# Initialise runner
runner = CliRunner()

# PARTICLES_XYZ: six particle positions (x, y, z) inside the 60^3 vesicle, 14 voxels from its centre
PARTICLES_XYZ = ((30, 30, 44), (30, 30, 16), (30, 44, 30), (30, 16, 30), (44, 30, 30), (16, 30, 30))

# _write_pair: write a vesicle as a matched segmentation/tomogram pair
def _write_pair(seg_dir: Path, raw_dir: Path, tomogram_id: str, *, voxel_size: float | None = None, seg_id: str | None = None) -> None:
    shape = (60, 60, 60)
    centre = np.array(shape) / 2.0
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1)
    distance = np.linalg.norm(grid - centre, axis=-1)
    segmentation = ((distance > 16.0) & (distance < 20.0)).astype(np.float32)
    tomogram = np.zeros(shape, dtype=np.float32)
    tomogram[segmentation > 0] = -1.0

    seg_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(seg_dir / f'{seg_id or tomogram_id}.mrc', overwrite=True) as mrc:
        mrc.set_data(segmentation)
    with mrcfile.new(raw_dir / f'{tomogram_id}.mrc', overwrite=True) as mrc:
        mrc.set_data(tomogram)
        if voxel_size is not None:
            mrc.voxel_size = voxel_size

# _write_picks: write a CSV picks file with PARTICLES_XYZ in each given tomogram
def _write_picks(path: Path, tomogram_ids: list[str], *, shift: float = 0.0) -> Path:
    rows = ['tomogram_id,x,y,z'] + [f'{tomogram_id},{x + shift},{y},{z}' for tomogram_id in tomogram_ids for x, y, z in PARTICLES_XYZ]
    path.write_text('\n'.join(rows) + '\n')
    return path

# _pick_args: CLI arguments for a pick run over the given files and directories
def _pick_args(tmp_path: Path, seg_dir: Path, raw_dir: Path, *picks_files: Path, extra: list[str] | None = None, voxel_size: str | None = '10.0') -> list[str]:
    return [
        '-d', str(tmp_path),
        'pick',
        *[str(path) for path in picks_files],
        '--seg-dir', str(seg_dir),
        '--raw-dir', str(raw_dir),
        '--out-dir', str(tmp_path),
        *(['--voxel-size-a', voxel_size] if voxel_size else []),
        *(extra or []),
    ]

# TestPickCommand: class containing tests for pick command
class TestPickCommand:
    # check `stamp pick` imports picks, infers orientations and writes outputs
    def test_pick_imports_picks_end_to_end(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'tomo000')
        picks = _write_picks(tmp_path / 'picks.csv', ['tomo000'])
        result = runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, picks))
        assert result.exit_code == 0, result.output

        output_dir = tmp_path / 'stamp' / 'pick'
        particle_set = ParticleSet.model_validate(json.loads((output_dir / 'particle_set.json').read_text()))
        assert len(particle_set.particles) == len(PARTICLES_XYZ)
        assert all(p.half_set is not None for p in particle_set.particles)
        assert all(p.orientation is not None for p in particle_set.particles)
        assert all(p.beam_angle_deviation_degrees is not None for p in particle_set.particles)
        assert {p.source_picker for p in particle_set.particles} == {'picks'}
        assert (output_dir / 'params.toml').exists()

    # check picks beyond --max-surface-distance-a are kept but get no orientation
    def test_pick_leaves_far_picks_unoriented(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'tomo000')
        picks = _write_picks(tmp_path / 'picks.csv', ['tomo000'])
        result = runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, picks, extra=['--max-surface-distance-a', '1']))
        assert result.exit_code == 0, result.output
        assert 'have no orientation' in result.output
        particle_set = ParticleSet.model_validate(json.loads((tmp_path / 'stamp' / 'pick' / 'particle_set.json').read_text()))
        assert all(p.orientation is None for p in particle_set.particles)

    # check pick ids absent from the segmentation directory are an error that names them
    def test_pick_rejects_unknown_tomogram_ids(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'tomo000')
        picks = _write_picks(tmp_path / 'picks.csv', ['tomo000', 'ghost'])
        result = runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, picks))
        assert result.exit_code != 0
        assert "'ghost'" in str(result.exception)

    # check a segmentation with no raw match is skipped
    def test_pick_skips_segmentations_without_raw_match(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'tomo000')
        # an extra segmentation with no matching raw tomogram.
        with mrcfile.new(seg_dir / 'orphan.mrc', overwrite=True) as mrc:
            mrc.set_data(np.zeros((60, 60, 60), dtype=np.float32))
        picks = _write_picks(tmp_path / 'picks.csv', ['tomo000'])
        result = runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, picks))
        assert 'No raw tomogram matching orphan, skipping' in result.output
        assert result.exit_code == 0, result.output

    # check a segmentation whose stem is the raw stem plus a suffix still matches
    def test_pick_matches_segmentation_by_stem_prefix(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'Position_1_stack_Vol', seg_id='Position_1_stack_Vol.denoised_segmented')
        # the picks file uses the raw-style id; the manifest id is the segmentation stem
        picks = _write_picks(tmp_path / 'picks.csv', ['Position_1_stack_Vol.denoised'])
        result = runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, picks))
        assert result.exit_code == 0, result.output
        assert 'No raw tomogram matching' not in result.output

    # check voxel size is read from the raw tomogram's MRC header when --voxel-size-a is omitted
    def test_pick_reads_voxel_size_from_header_when_omitted(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'tomo000', voxel_size=10.0)
        picks = _write_picks(tmp_path / 'picks.csv', ['tomo000'])
        result = runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, picks, voxel_size=None))
        assert result.exit_code == 0, result.output

    # check mixed voxel sizes across tomograms warn and fall back to the most common
    def test_pick_warns_and_uses_most_common_voxel_size(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        ids = ['tomo000', 'tomo001', 'tomo002']
        for tomogram_id, voxel_size in zip(ids, (10.0, 10.0, 14.0)):
            _write_pair(seg_dir, raw_dir, tomogram_id, voxel_size=voxel_size)
        picks = _write_picks(tmp_path / 'picks.csv', ids)
        result = runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, picks, voxel_size=None))
        assert result.exit_code == 0, result.output
        assert 'Multiple voxel sizes were found' in result.output

    # check two picks files with the intersection rule yield picks attributed to both pickers
    def test_pick_intersection_joins_source_pickers(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'tomo000')
        first = _write_picks(tmp_path / 'first.csv', ['tomo000'])
        second = _write_picks(tmp_path / 'second.csv', ['tomo000'], shift=2.0)  # within the 15 voxel default threshold
        result = runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, first, second, extra=['--consensus-rule', 'intersection']))
        assert result.exit_code == 0, result.output
        particle_set = ParticleSet.model_validate(json.loads((tmp_path / 'stamp' / 'pick' / 'particle_set.json').read_text()))
        assert len(particle_set.particles) == len(PARTICLES_XYZ)
        assert {p.source_picker for p in particle_set.particles} == {'first+second'}

    # check --name overrides the file stem and duplicate names are rejected
    def test_pick_names(self, tmp_path: Path) -> None:
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        _write_pair(seg_dir, raw_dir, 'tomo000')
        picks = _write_picks(tmp_path / 'picks.csv', ['tomo000'])
        result = runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, picks, extra=['--name', 'mine']))
        assert result.exit_code == 0, result.output
        particle_set = ParticleSet.model_validate(json.loads((tmp_path / 'stamp' / 'pick' / 'particle_set.json').read_text()))
        assert {p.source_picker for p in particle_set.particles} == {'mine'}

        other = tmp_path / 'other'
        other.mkdir()
        twin = _write_picks(other / 'picks.csv', ['tomo000'])
        result = runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, picks, twin))
        assert result.exit_code != 0
        assert 'must be unique' in str(result.exception)

    # check an interrupted pick resumes, orienting only the tomograms that had not finished
    def test_pick_resumes_after_interruption(self, tmp_path: Path, monkeypatch) -> None:
        import stamp.commands.pick as pickfuncs
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        ids = ['tomo000', 'tomo001', 'tomo002']
        for tomogram_id in ids:
            _write_pair(seg_dir, raw_dir, tomogram_id)
        picks = _write_picks(tmp_path / 'picks.csv', ids)
        real = pickfuncs.infer_orientations
        oriented: list[str] = []
        state = {'fail': True}

        # flaky: fail on tomo001 until told otherwise, else infer normally
        def flaky(tomogram_picks, manifests, **kwargs):
            tomogram_id = tomogram_picks[0].tomogram_id
            if state['fail'] and tomogram_id == 'tomo001':
                raise RuntimeError('boom')
            oriented.append(tomogram_id)
            return real(tomogram_picks, manifests, **kwargs)

        monkeypatch.setattr(pickfuncs, 'infer_orientations', flaky)
        args = _pick_args(tmp_path, seg_dir, raw_dir, picks)
        assert runner.invoke(stamp_app, args).exit_code != 0
        assert oriented == ['tomo000']  # picks are grouped in file order

        state['fail'] = False
        oriented.clear()
        result = runner.invoke(stamp_app, args)
        assert result.exit_code == 0, result.output
        assert oriented == ['tomo001', 'tomo002']

    # check a changed picks file discards the checkpoint
    def test_pick_restarts_when_picks_change(self, tmp_path: Path, monkeypatch) -> None:
        import stamp.commands.pick as pickfuncs
        seg_dir, raw_dir = tmp_path / 'seg', tmp_path / 'raw'
        ids = ['tomo000', 'tomo001']
        for tomogram_id in ids:
            _write_pair(seg_dir, raw_dir, tomogram_id)
        picks = _write_picks(tmp_path / 'picks.csv', ids)
        real = pickfuncs.infer_orientations
        oriented: list[str] = []
        state = {'fail': True}

        # flaky: fail on tomo001 until told otherwise, else infer normally
        def flaky(tomogram_picks, manifests, **kwargs):
            tomogram_id = tomogram_picks[0].tomogram_id
            if state['fail'] and tomogram_id == 'tomo001':
                raise RuntimeError('boom')
            oriented.append(tomogram_id)
            return real(tomogram_picks, manifests, **kwargs)

        monkeypatch.setattr(pickfuncs, 'infer_orientations', flaky)
        runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, picks))
        state['fail'] = False
        oriented.clear()
        _write_picks(picks, ids, shift=1.0)  # same path, new content
        runner.invoke(stamp_app, _pick_args(tmp_path, seg_dir, raw_dir, picks))
        assert oriented == ['tomo000', 'tomo001']


# TestPickOrientationPrior: class containing tests for --orientation-prior
class TestPickOrientationPrior:
    # check the none prior needs no segmentation, drops reported orientations and tags the particle set
    def test_none_prior_without_segmentation(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / 'raw'
        _write_pair(tmp_path / 'unused', raw_dir, 'tomo000')
        picks = tmp_path / 'picks.csv'
        picks.write_text('tomogram_id,x,y,z,qw,qx,qy,qz\n' + '\n'.join(f'tomo000,{x},{y},{z},1,0,0,0' for x, y, z in PARTICLES_XYZ) + '\n')
        result = runner.invoke(stamp_app, ['-d', str(tmp_path), 'pick', str(picks), '--raw-dir', str(raw_dir), '--out-dir', str(tmp_path), '--voxel-size-a', '10.0', '--orientation-prior', 'none'])
        assert result.exit_code == 0, result.output
        particle_set = ParticleSet.model_validate(json.loads((tmp_path / 'stamp' / 'pick' / 'particle_set.json').read_text()))
        assert particle_set.orientation_prior.value == 'none'
        assert len(particle_set.particles) == len(PARTICLES_XYZ) and all(p.orientation is None for p in particle_set.particles)

    # check the axis prior keeps the supplied axis and refuses picks without one
    def test_axis_prior_requires_orientations(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / 'raw'
        _write_pair(tmp_path / 'unused', raw_dir, 'tomo000')
        with_axis = tmp_path / 'with_axis.csv'
        with_axis.write_text('tomogram_id,x,y,z,qw,qx,qy,qz\n' + '\n'.join(f'tomo000,{x},{y},{z},0.7071068,0,0.7071068,0' for x, y, z in PARTICLES_XYZ) + '\n')
        base = ['-d', str(tmp_path), 'pick', '--raw-dir', str(raw_dir), '--out-dir', str(tmp_path), '--voxel-size-a', '10.0', '--orientation-prior', 'axis']
        ok = runner.invoke(stamp_app, [*base[:2], base[2], str(with_axis), *base[3:]])
        assert ok.exit_code == 0, ok.output
        particle_set = ParticleSet.model_validate(json.loads((tmp_path / 'stamp' / 'pick' / 'particle_set.json').read_text()))
        assert particle_set.orientation_prior.value == 'axis' and all(p.orientation is not None for p in particle_set.particles)
        without = _write_picks(tmp_path / 'without.csv', ['tomo000'])
        bad = runner.invoke(stamp_app, [*base[:2], base[2], str(without), *base[3:]])
        assert bad.exit_code != 0 and 'axis needs an orientation' in bad.output

    # check the normal prior still demands a segmentation directory
    def test_normal_prior_requires_segmentation(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / 'raw'
        _write_pair(tmp_path / 'seg', raw_dir, 'tomo000')
        picks = _write_picks(tmp_path / 'picks.csv', ['tomo000'])
        result = runner.invoke(stamp_app, ['-d', str(tmp_path), 'pick', str(picks), '--raw-dir', str(raw_dir), '--out-dir', str(tmp_path), '--voxel-size-a', '10.0'])
        assert result.exit_code != 0 and '--seg-dir is required' in result.output
