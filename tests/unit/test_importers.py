'''
STAMP: unit tests for the picks importer
'''

# Import external dependencies
import json, numpy as np, pandas as pd, pytest, starfile
from pathlib import Path

# Import internal STAMP objects
from stamp.picking.geometry import quaternion_from_reference_to
from stamp.picking.importers import infer_orientations, map_tomogram_ids, read_picks
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.picks import RawPick
from stamp.utils.errors import StampValidationError
from stamp.utils.io import new_mrc

# _manifest: a manifest with placeholder paths
def _manifest(tomogram_id: str) -> TomogramManifest:
    return TomogramManifest(tomogram_id=tomogram_id, segmentation_path=Path(f'{tomogram_id}.mrc'), raw_tomogram_path=Path('raw.mrc'), voxel_size_angstrom=10.0)

# _raw: a RawPick for id mapping tests
def _raw(tomogram_id: str) -> RawPick:
    return RawPick(tomogram_id=tomogram_id, position=(0.0, 0.0, 0.0), source_picker='mine')

# _slab_manifest: flat one-voxel membrane at x=30 in a 60^3 volume, voxel size 10 Å
def _slab_manifest(tmp_path: Path) -> TomogramManifest:
    segmentation = np.zeros((60, 60, 60), dtype=np.float32)
    segmentation[:, :, 30] = 1.0
    new_mrc(tmp_path / 't00.mrc', segmentation)
    new_mrc(tmp_path / 't00_raw.mrc', np.zeros_like(segmentation))
    return TomogramManifest(tomogram_id='t00', segmentation_path=tmp_path / 't00.mrc', raw_tomogram_path=tmp_path / 't00_raw.mrc', voxel_size_angstrom=10.0)

# _at: a RawPick at an (x, y, z) position in t00
def _at(position: tuple[float, float, float], **fields) -> RawPick:
    return RawPick(tomogram_id='t00', position=position, source_picker='mine', **fields)

# TestApplyPrior: class containing tests for apply_prior in src/stamp/picking/importers.py
class TestApplyPrior:
    # test_apply_prior_per_prior: normal passes picks through, axis demands orientations, none strips orientation and membrane fields
    def test_apply_prior_per_prior(self) -> None:
        from stamp.picking.importers import apply_prior
        from stamp.schemas.particles import OrientationPrior
        from stamp.schemas.picks import RawPick
        bare = RawPick(tomogram_id='t', position=(1.0, 2.0, 3.0), source_picker='a')
        oriented = bare.model_copy(update={'orientation': (1.0, 0.0, 0.0, 0.0), 'beam_angle_deviation_degrees': 10.0, 'vesicle_id': 'v1'})
        assert apply_prior([bare], OrientationPrior.NORMAL) == [bare]
        assert apply_prior([oriented], OrientationPrior.AXIS) == [oriented]
        with pytest.raises(StampValidationError, match='1 of 2'):
            apply_prior([bare, oriented], OrientationPrior.AXIS)
        stripped = apply_prior([oriented], OrientationPrior.NONE)[0]
        assert stripped.orientation is None and stripped.beam_angle_deviation_degrees is None and stripped.vesicle_id is None and stripped.position == oriented.position

# TestReadPicks: class containing tests for read_picks
class TestReadPicks:
    def test_csv_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y,z,confidence\nt00,1.5,2.5,3.5,0.9\nt01,4,5,6,\n')
        picks = read_picks(path, source_picker='mine')
        assert [p.tomogram_id for p in picks] == ['t00', 't01']
        assert picks[0].position == (1.5, 2.5, 3.5)
        assert picks[0].confidence == 0.9 and picks[1].confidence is None
        assert all(p.source_picker == 'mine' and p.orientation is None for p in picks)

    def test_csv_quaternion_is_kept(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y,z,qw,qx,qy,qz\nt00,1,2,3,1,0,0,0\n')
        assert read_picks(path, source_picker='mine')[0].orientation == (1.0, 0.0, 0.0, 0.0)

    def test_csv_partial_quaternion_columns_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y,z,qw\nt00,1,2,3,1\n')
        with pytest.raises(StampValidationError, match='all of'):
            read_picks(path, source_picker='mine')

    def test_csv_missing_column_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y\nt00,1,2\n')
        with pytest.raises(StampValidationError, match="'z'"):
            read_picks(path, source_picker='mine')

    def test_csv_bad_row_reports_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y,z\nt00,1,2,3\nt00,oops,2,3\n')
        with pytest.raises(StampValidationError, match='line 3'):
            read_picks(path, source_picker='mine')

    def test_csv_non_unit_quaternion_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.csv'
        path.write_text('tomogram_id,x,y,z,qw,qx,qy,qz\nt00,1,2,3,2,0,0,0\n')
        with pytest.raises(StampValidationError, match='line 2'):
            read_picks(path, source_picker='mine')

    def test_star_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.star'
        table = pd.DataFrame({
            'rlnTomoName': ['t00', 't00'],
            'rlnCoordinateX': [10.0, 20.0],
            'rlnCoordinateY': [11.0, 21.0],
            'rlnCoordinateZ': [12.0, 22.0],
            'rlnAutopickFigureOfMerit': [0.5, 0.7],
        })
        starfile.write({'particles': table}, path)
        picks = read_picks(path, source_picker='relion')
        assert [p.position for p in picks] == [(10.0, 11.0, 12.0), (20.0, 21.0, 22.0)]
        assert [p.confidence for p in picks] == [0.5, 0.7]

    def test_star_without_coordinates_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.star'
        starfile.write({'particles': pd.DataFrame({'rlnTomoName': ['t00'], 'rlnOriginXAngst': [1.0]})}, path)
        with pytest.raises(StampValidationError, match='rlnCoordinateX'):
            read_picks(path, source_picker='relion')

    def test_json_overwrites_source_picker(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.json'
        path.write_text(json.dumps([{'tomogram_id': 't00', 'position': [1, 2, 3], 'source_picker': 'other'}]))
        assert read_picks(path, source_picker='mine')[0].source_picker == 'mine'

    def test_jsonl_reads_one_record_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / 'picks.jsonl'
        path.write_text('\n'.join(json.dumps({'tomogram_id': 't00', 'position': [i, 0, 0], 'source_picker': 'x'}) for i in range(3)))
        assert len(read_picks(path, source_picker='mine')) == 3

    def test_unknown_suffix_and_empty_file_are_rejected(self, tmp_path: Path) -> None:
        (tmp_path / 'picks.txt').write_text('')
        with pytest.raises(StampValidationError, match='unsupported'):
            read_picks(tmp_path / 'picks.txt', source_picker='mine')
        (tmp_path / 'empty.csv').write_text('tomogram_id,x,y,z\n')
        with pytest.raises(StampValidationError, match='no picks'):
            read_picks(tmp_path / 'empty.csv', source_picker='mine')

# TestMapTomogramIds: class containing tests for map_tomogram_ids
class TestMapTomogramIds:
    def test_exact_match_is_unchanged(self) -> None:
        assert map_tomogram_ids([_raw('t00')], [_manifest('t00')])[0].tomogram_id == 't00'

    def test_segmented_suffix_is_ignored_on_either_side(self) -> None:
        manifests = [_manifest('a_segmented'), _manifest('b')]
        mapped = map_tomogram_ids([_raw('a'), _raw('b_segmented')], manifests)
        assert [p.tomogram_id for p in mapped] == ['a_segmented', 'b']

    def test_unknown_ids_are_listed(self) -> None:
        with pytest.raises(StampValidationError, match="'nope'"):
            map_tomogram_ids([_raw('t00'), _raw('nope')], [_manifest('t00')])

# TestInferOrientations: class containing tests for infer_orientations
class TestInferOrientations:
    def test_normal_points_from_surface_towards_the_pick(self, tmp_path: Path) -> None:
        manifest = _slab_manifest(tmp_path)
        right, left = infer_orientations([_at((35.0, 20.0, 20.0)), _at((25.0, 20.0, 20.0))], [manifest])
        assert np.allclose(right.orientation, quaternion_from_reference_to(np.array([1.0, 0.0, 0.0])), atol=1e-6)
        assert np.allclose(left.orientation, quaternion_from_reference_to(np.array([-1.0, 0.0, 0.0])), atol=1e-6)
        assert right.beam_angle_deviation_degrees == pytest.approx(0.0, abs=1e-6)  # normal in the xy plane

    def test_picks_beyond_max_distance_stay_unoriented(self, tmp_path: Path) -> None:
        manifest = _slab_manifest(tmp_path)
        near, far = infer_orientations([_at((32.0, 20.0, 20.0)), _at((50.0, 20.0, 20.0))], [manifest], max_surface_distance_angstrom=50.0)
        assert near.orientation is not None and far.orientation is None

    def test_supplied_quaternion_is_kept_and_sets_beam_angle(self, tmp_path: Path) -> None:
        manifest = _slab_manifest(tmp_path)
        (pick,) = infer_orientations([_at((35.0, 20.0, 20.0), orientation=(1.0, 0.0, 0.0, 0.0))], [manifest])
        assert pick.orientation == (1.0, 0.0, 0.0, 0.0)
        assert pick.beam_angle_deviation_degrees == pytest.approx(90.0)  # +z is beam-aligned

    def test_vesicle_id_is_filled_from_labels(self, tmp_path: Path) -> None:
        manifest = _slab_manifest(tmp_path)
        labels = np.zeros((60, 60, 60), dtype=np.int16)
        labels[:, :, 29:32] = 3
        new_mrc(tmp_path / 't00_labelled.mrc', labels)
        (pick,) = infer_orientations([_at((35.0, 20.0, 20.0))], [manifest], vesicle_labels_by_tomogram={'t00': tmp_path / 't00_labelled.mrc'})
        assert pick.vesicle_id == 't00:v0003'

    def test_empty_segmentation_leaves_picks_untouched(self, tmp_path: Path) -> None:
        manifest = _slab_manifest(tmp_path)
        new_mrc(manifest.segmentation_path, np.zeros((60, 60, 60), dtype=np.float32))
        (pick,) = infer_orientations([_at((35.0, 20.0, 20.0))], [manifest])
        assert pick.orientation is None
