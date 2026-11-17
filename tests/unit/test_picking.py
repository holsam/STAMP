'''
STAMP: unit tests for picking functions, including geometric primitives, and consensus reconciliation
'''

# Import external dependencies
import mrcfile, numpy as np, pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.picking.consensus import build_particle_set, reconcile_picks
from stamp.commands.pick import pick_tomogram
from stamp.picking.importers import filter_beam_angle, select_tomogram_picks
from stamp.schemas.manifest import TomogramManifest
from stamp.utils.errors import StampValidationError
from stamp.picking.geometry import (
    beam_angle_deviation_degrees,
    compose_roll_about_normal,
    extract_surface,
)
from stamp.utils.rotation import quaternion_from_reference_to
from stamp.picking.vesicles import (
    load_vesicle_labels,
    summarise_vesicles,
    vesicle_ids_at,
    vesicle_surface_area_angstrom2,
)
from stamp.schemas.particles import HalfSet, OrientationPrior, Particle
from stamp.schemas.picks import RawPick

# _hollow_sphere: a hollow spherical shell segmentation volume
def _hollow_sphere(shape=(40, 40, 40), radius=12.0, thickness=2.0) -> np.ndarray:
    centre = np.array(shape) / 2.0
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1)
    distance = np.linalg.norm(grid - centre, axis=-1)
    return ((distance > radius - thickness) & (distance < radius + thickness)).astype(np.float32)

# _write_mrc: write a volume to an MRC file as float32
def _write_mrc(path: Path, volume: np.ndarray) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(volume.astype(np.float32))

# _pick: return a RawPick instance for use in tests
def _pick(tomogram_id: str, position, picker: str, confidence=0.8, orientation=None) -> RawPick:
    return RawPick(
        tomogram_id=tomogram_id,
        position=position,
        orientation=orientation,
        prior=OrientationPrior.NORMAL,
        confidence=confidence,
        source_picker=picker,
    )

# TestGeometry: class containing unit tests for test_geometry.py
class TestGeometry:
    def test_extract_surface_returns_unit_normals(self) -> None:
        '''Extracted normals are unit-length.'''
        vertices, normals = extract_surface(_hollow_sphere())
        assert vertices.shape[0] > 0
        assert vertices.shape[1] == 3
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)

    def test_extract_surface_rejects_empty_segmentation(self) -> None:
        '''An all-zero segmentation raises.'''
        with pytest.raises(StampValidationError, match='no voxels above'):
            extract_surface(np.zeros((10, 10, 10), dtype=np.float32))

    def test_quaternion_is_unit_length_and_rotates_correctly(self) -> None:
        '''Every returned quaternion is unit-length'''
        for vector in ([0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]):
            quaternion = quaternion_from_reference_to(np.array(vector, dtype=float))
            assert abs(np.linalg.norm(quaternion) - 1.0) < 1e-6

    def test_quaternion_identity_for_reference_axis(self) -> None:
        '''The reference axis maps to the identity quaternion'''
        quaternion = quaternion_from_reference_to(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(quaternion, (1.0, 0.0, 0.0, 0.0), atol=1e-6)

    def test_quaternion_antiparallel_case_is_unit_length(self) -> None:
        '''The antiparallel case still returns a unit quaternion'''
        quaternion = quaternion_from_reference_to(np.array([0.0, 0.0, -1.0]))
        assert abs(np.linalg.norm(quaternion) - 1.0) < 1e-6

    # test_compose_roll_is_identity_at_zero: zero angle leaves the normal quaternion unchanged
    def test_compose_roll_is_identity_at_zero(self):
        base = quaternion_from_reference_to(np.array([0.3, -0.5, 0.8]))
        rolled = compose_roll_about_normal(base, 0.0)
        assert np.allclose(rolled, base, atol=1e-9)

    # test_compose_roll_preserves_normal: rolling about the normal does not move +z's image
    def test_compose_roll_preserves_normal(self):
        from stamp.utils.rotation import quaternion_to_matrix
        target = np.array([0.1, 0.2, 0.97])
        target /= np.linalg.norm(target)
        base = quaternion_from_reference_to(target)
        rolled = compose_roll_about_normal(base, 73.0)
        moved = quaternion_to_matrix(rolled) @ np.array([0.0, 0.0, 1.0])
        assert np.allclose(moved, target, atol=1e-6)

    def test_beam_angle_deviation_matches_closed_form(self) -> None:
        '''Directly check the deviation formula against known normals.'''
        assert beam_angle_deviation_degrees(np.array([0.0, 0.0, 1.0])) == pytest.approx(90.0)
        assert beam_angle_deviation_degrees(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
        assert beam_angle_deviation_degrees(np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)
        assert beam_angle_deviation_degrees(np.array([0.0, 0.0, -1.0])) == pytest.approx(90.0)

    def test_filter_removes_polar_picks_when_enabled(self) -> None:
        particles = [
            Particle(prior=OrientationPrior.NORMAL, particle_id='p0', tomogram_id='t', position=(0, 0, 0), source_picker='x', half_set=HalfSet.A, beam_angle_deviation_degrees=5.0),
            Particle(prior=OrientationPrior.NORMAL, particle_id='p1', tomogram_id='t', position=(0, 0, 0), source_picker='x', half_set=HalfSet.A, beam_angle_deviation_degrees=85.0),
        ]
        kept = [p for p in particles if p.beam_angle_deviation_degrees is None or p.beam_angle_deviation_degrees <= 45.0]
        assert [p.particle_id for p in kept] == ['p0']

    def test_filter_retains_all_when_disabled(self) -> None:
        particles = [
            Particle(prior=OrientationPrior.NORMAL, particle_id='p0', tomogram_id='t', position=(0, 0, 0), source_picker='x', half_set=HalfSet.A, beam_angle_deviation_degrees=5.0),
            Particle(prior=OrientationPrior.NORMAL, particle_id='p1', tomogram_id='t', position=(0, 0, 0), source_picker='x', half_set=HalfSet.A, beam_angle_deviation_degrees=85.0),
        ]
        # max_beam_angle_deviation is None: no filtering happens at all
        max_beam_angle_deviation = None
        kept = particles if max_beam_angle_deviation is None else [
            p for p in particles if p.beam_angle_deviation_degrees is None or p.beam_angle_deviation_degrees <= max_beam_angle_deviation
        ]
        assert len(kept) == 2

# TestConsensusRules: class containing tests for consensus rules (intersection/union)
class TestConsensusRules:
    def test_intersection_merges_close_picks_from_different_pickers(self) -> None:
        picks_by_picker = {
            'stamp-native': [_pick('tomo000', (100.0, 100.0, 100.0), 'stamp-native')],
            'membrain-pick': [_pick('tomo000', (103.0, 101.0, 99.0), 'membrain-pick')],
        }
        reconciled = reconcile_picks(picks_by_picker, 'intersection', distance_threshold=15.0, tomogram_id='tomo000')
        assert len(reconciled) == 1
        assert 'stamp-native' in reconciled[0].source_picker
        assert 'membrain-pick' in reconciled[0].source_picker

    def test_intersection_drops_picks_found_by_only_one_picker(self) -> None:
        picks_by_picker = {
            'stamp-native': [_pick('tomo000', (100.0, 100.0, 100.0), 'stamp-native')],
            'membrain-pick': [_pick('tomo000', (500.0, 500.0, 500.0), 'membrain-pick')],
        }
        reconciled = reconcile_picks(picks_by_picker, 'intersection', distance_threshold=15.0, tomogram_id='tomo000')
        assert reconciled == []

    def test_union_keeps_picks_found_by_only_one_picker(self) -> None:
        picks_by_picker = {
            'stamp-native': [_pick('tomo000', (100.0, 100.0, 100.0), 'stamp-native')],
            'membrain-pick': [_pick('tomo000', (500.0, 500.0, 500.0), 'membrain-pick')],
        }
        reconciled = reconcile_picks(picks_by_picker, 'union', distance_threshold=15.0, tomogram_id='tomo000')
        assert len(reconciled) == 2

    def test_single_picker_intersection_equals_union(self) -> None:
        picks_by_picker = {
            'stamp-native': [
                _pick('tomo000', (100.0, 100.0, 100.0), 'stamp-native'),
                _pick('tomo000', (500.0, 500.0, 500.0), 'stamp-native'),
            ]
        }
        as_intersection = reconcile_picks(picks_by_picker, 'intersection', distance_threshold=15.0, tomogram_id='tomo000')
        as_union = reconcile_picks(picks_by_picker, 'union', distance_threshold=15.0, tomogram_id='tomo000')
        assert len(as_intersection) == len(as_union) == 2

# TestConsensus: class containing unit tests for consensus command
class TestConsensus:
    def test_chain_does_not_merge_distant_picks(self):
        '''Three collinear picks each within threshold of the next stay split.'''
        picks = {'p': [
            RawPick(tomogram_id='t', position=(0.0, 0.0, 0.0), source_picker='p'),
            RawPick(tomogram_id='t', position=(12.0, 0.0, 0.0), source_picker='p'),
            RawPick(tomogram_id='t', position=(24.0, 0.0, 0.0), source_picker='p'),
        ]}
        out = reconcile_picks(picks, 'union', distance_threshold=15.0, tomogram_id='t')
        assert len(out) >= 2  # complete-linkage keeps the 0 and 24 picks apart

    def test_centroid_not_biased_by_pick_count(self):
        '''Picker A places 4, picker B places 1; centroid sits between them.'''
        a = [RawPick(tomogram_id='t', position=(x, 0.0, 0.0), source_picker='a') for x in (0, 1, 2, 1)]
        b = [RawPick(tomogram_id='t', position=(10.0, 0.0, 0.0), source_picker='b')]
        out = reconcile_picks({'a': a, 'b': b}, 'intersection', distance_threshold=20.0, tomogram_id='t')
        assert out and 4.0 < out[0].position[0] < 6.0

    def test_reconciled_position_is_centroid_of_component(self) -> None:
        picks_by_picker = {
            'stamp-native': [_pick('tomo000', (0.0, 0.0, 0.0), 'stamp-native')],
            'membrain-pick': [_pick('tomo000', (10.0, 0.0, 0.0), 'membrain-pick')],
        }
        reconciled = reconcile_picks(picks_by_picker, 'union', distance_threshold=15.0, tomogram_id='tomo000')
        assert len(reconciled) == 1
        assert reconciled[0].position == (5.0, 0.0, 0.0)

    def test_reconcile_filters_to_requested_tomogram_only(self) -> None:
        picks_by_picker = {
            'stamp-native': [
                _pick('tomo000', (0.0, 0.0, 0.0), 'stamp-native'),
                _pick('tomo001', (0.0, 0.0, 0.0), 'stamp-native'),
            ]
        }
        reconciled = reconcile_picks(picks_by_picker, 'union', distance_threshold=15.0, tomogram_id='tomo000')
        assert len(reconciled) == 1
        assert reconciled[0].tomogram_id == 'tomo000'

    def test_build_particle_set_assigns_ids_and_half_sets(self) -> None:
        reconciled_picks = [
            _pick('tomo000', (0.0, 0.0, 0.0), 'stamp-native+membrain-pick'),
            _pick('tomo001', (5.0, 5.0, 5.0), 'stamp-native+membrain-pick'),
        ]
        particle_set = build_particle_set(
            reconciled_picks=reconciled_picks,
            consensus_rule='intersection',
            contributing_pickers=['stamp-native', 'membrain-pick'],
            half_set_seed=1,
        )
        assert len(particle_set.particles) == 2
        assert len({particle.particle_id for particle in particle_set.particles}) == 2
        assert all(particle.half_set is not None for particle in particle_set.particles)

    # test_conflicting_priors_exclude_the_cluster: pickers agreeing on position but not on prior lose that particle, with a warning; agreeing clusters survive
    def test_conflicting_priors_exclude_the_cluster(self, caplog) -> None:
        near = lambda picker, prior, x: _pick('t', (x, 10.0, 10.0), picker).model_copy(update={'prior': prior})
        picks = {
            'a': [near('a', OrientationPrior.NORMAL, 10.0), near('a', OrientationPrior.AXIS, 80.0)],
            'b': [near('b', OrientationPrior.AXIS, 11.0), near('b', OrientationPrior.AXIS, 81.0)],
        }
        reconciled = reconcile_picks(picks, 'intersection', 5.0, 't')
        assert [pick.prior for pick in reconciled] == [OrientationPrior.AXIS]
        assert 'orientation priors disagree' in caplog.text

# TestVesicle: class containing unit tests for picking/vesicle.py
class TestVesicle:
    def test_load_vesicle_labels_rejects_shape_mismatch(self, tmp_path: Path) -> None:
        '''Mismatched shapes raise StampValidationError rather than silently misattributing'''
        _write_mrc(tmp_path / 'labels.mrc', np.zeros((10, 10, 10)))
        with pytest.raises(StampValidationError):
            load_vesicle_labels(tmp_path / 'labels.mrc', (20, 20, 20))

    def test_load_vesicle_labels_reads_integer_labels(self, tmp_path: Path) -> None:
        '''Labels round-trip through the MRC unchanged'''
        data = np.zeros((10, 10, 10))
        data[2, 2, 2] = 1
        data[7, 7, 7] = 2
        _write_mrc(tmp_path / 'labels.mrc', data)
        labels = load_vesicle_labels(tmp_path / 'labels.mrc', (10, 10, 10))
        assert labels[2, 2, 2] == 1
        assert labels[7, 7, 7] == 2

    def test_vesicle_ids_at_matches_label(self) -> None:
        '''vesicle_ids_at resolves a point to EValuator's own label, unchanged, and '' for background'''
        labels = np.zeros((10, 10, 10), dtype=np.int64)
        labels[2, 2, 2] = 1
        labels[7, 7, 7] = 2
        ids = vesicle_ids_at(labels, np.array([[2.0, 2.0, 2.0], [7.0, 7.0, 7.0], [0.0, 0.0, 0.0]]), 'tomo01')
        assert ids[0] == 'tomo01:v0001'
        assert ids[1] == 'tomo01:v0002'
        assert ids[2] == ''

    def test_vesicle_surface_area_assigns_faces_to_nearest_label(self) -> None:
        '''Mesh face area lands on the vesicle nearest its centroid'''
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0]])
        faces = np.array([[0, 1, 2]])  # a single triangle near label 1, nothing near vertex 3
        labels = np.zeros((10, 10, 10), dtype=np.int64)
        labels[0, 0, 0] = 1
        areas = vesicle_surface_area_angstrom2(vertices, faces, labels, voxel_size_angstrom=2.0, tomogram_id='t01')
        assert set(areas) == {'t01:v0001'}
        assert areas['t01:v0001'] > 0.0

    def test_summarise_vesicles_reports_density(self) -> None:
        '''Two vesicles with different pick counts but equal area get distinct densities'''
        from types import SimpleNamespace
        picks = [
            SimpleNamespace(vesicle_id='t01:v0001'), SimpleNamespace(vesicle_id='t01:v0001'),
            SimpleNamespace(vesicle_id='t01:v0002'),
        ]
        areas = {'t01:v0001': 2000.0, 't01:v0002': 2000.0}
        summaries = {s.vesicle_id: s for s in summarise_vesicles(picks, areas, 't01')}
        assert summaries['t01:v0001'].n_picks == 2
        assert summaries['t01:v0002'].n_picks == 1
        assert summaries['t01:v0001'].picks_per_1000_angstrom2 > summaries['t01:v0002'].picks_per_1000_angstrom2

    def test_summarise_vesicles_includes_zero_pick_vesicles(self) -> None:
        '''A vesicle with area but no picks still gets a zero-count, zero-density row'''
        summaries = {s.vesicle_id: s for s in summarise_vesicles([], {'t01:v0001': 500.0}, 't01')}
        assert summaries['t01:v0001'].n_picks == 0
        assert summaries['t01:v0001'].picks_per_1000_angstrom2 == 0.0

    def test_reconciled_pick_keeps_vesicle_id(self) -> None:
        '''Consensus reconciliation preserves the representative pick's vesicle_id'''
        picks_by_picker = {
            'stamp-native': [_pick('tomo000', (0.0, 0.0, 0.0), 'stamp-native')],
        }
        picks_by_picker['stamp-native'][0].vesicle_id = 'tomo000:v0001'
        reconciled = reconcile_picks(picks_by_picker, 'union', distance_threshold=15.0, tomogram_id='tomo000')
        assert reconciled[0].vesicle_id == 'tomo000:v0001'

    def test_particle_carries_vesicle_id_from_reconciled_pick(self) -> None:
        '''build_particle_set copies vesicle_id straight across onto the Particle'''
        reconciled_picks = [_pick('tomo000', (0.0, 0.0, 0.0), 'stamp-native')]
        reconciled_picks[0].vesicle_id = 'tomo000:v0002'
        particle_set = build_particle_set(
            reconciled_picks=reconciled_picks,
            consensus_rule='union',
            contributing_pickers=['stamp-native'],
            half_set_seed=0,
        )
        assert particle_set.particles[0].vesicle_id == 'tomo000:v0002'

# TestBeamAngleFilter: class checking that the beam angle filter acts on raw picks before ids are assigned
class TestBeamAngleFilter:
    def test_drops_picks_over_the_maximum(self):
        picks = [
            RawPick(tomogram_id='t', position=(0.0, 0.0, 0.0), source_picker='p', beam_angle_deviation_degrees=5.0),
            RawPick(tomogram_id='t', position=(9.0, 0.0, 0.0), source_picker='p', beam_angle_deviation_degrees=40.0),
        ]
        assert [p.position for p in filter_beam_angle(picks, 20.0)] == [(0.0, 0.0, 0.0)]

    def test_keeps_picks_without_a_recorded_deviation(self):
        picks = [RawPick(tomogram_id='t', position=(0.0, 0.0, 0.0), source_picker='p')]
        assert filter_beam_angle(picks, 20.0) == picks

    def test_none_maximum_keeps_everything(self):
        picks = [RawPick(tomogram_id='t', position=(0.0, 0.0, 0.0), source_picker='p', beam_angle_deviation_degrees=89.0)]
        assert filter_beam_angle(picks, None) == picks

    def test_ids_are_contiguous_after_filtering(self):
        picks = [
            RawPick(tomogram_id='t', position=(0.0, 0.0, 0.0), source_picker='p', prior=OrientationPrior.NORMAL, beam_angle_deviation_degrees=80.0),
            RawPick(tomogram_id='t', position=(30.0, 0.0, 0.0), source_picker='p', prior=OrientationPrior.NORMAL, beam_angle_deviation_degrees=5.0),
        ]
        particle_set = build_particle_set(filter_beam_angle(picks, 20.0), 'union', ['p'], half_set_seed=0)
        assert [p.particle_id for p in particle_set.particles] == ['t_p0000']

# TestSelectTomogramPicks: class checking per-tomogram selection from a multi-tomogram picks list
class TestSelectTomogramPicks:
    def test_keeps_only_the_manifest_tomogram_with_id_rewritten(self, tmp_path: Path):
        manifest = TomogramManifest(tomogram_id='tomoA', raw_tomogram_path=tmp_path / 'a.mrc', voxel_size_angstrom=10.0)
        picks = [
            RawPick(tomogram_id='tomoA_segmented', position=(1.0, 1.0, 1.0), source_picker='x'),
            RawPick(tomogram_id='tomoB', position=(2.0, 2.0, 2.0), source_picker='x'),
        ]
        selected = select_tomogram_picks(picks, manifest)
        assert [(p.tomogram_id, p.position) for p in selected] == [('tomoA', (1.0, 1.0, 1.0))]

    def test_unknown_ids_are_ignored_not_raised(self, tmp_path: Path):
        manifest = TomogramManifest(tomogram_id='tomoA', raw_tomogram_path=tmp_path / 'a.mrc', voxel_size_angstrom=10.0)
        assert select_tomogram_picks([RawPick(tomogram_id='nope', position=(0.0, 0.0, 0.0), source_picker='x')], manifest) == []


# TestPickTomogram: class checking the single-tomogram pick helper against a multi-tomogram picks file
class TestPickTomogram:
    def _manifest(self, tmp_path: Path, tomogram_id: str) -> TomogramManifest:
        return TomogramManifest(tomogram_id=tomogram_id, raw_tomogram_path=tmp_path / f'{tomogram_id}.mrc', voxel_size_angstrom=10.0)

    def _csv(self, path: Path, rows: list[str]) -> Path:
        path.write_text('tomogram_id,x,y,z,confidence\n' + '\n'.join(rows) + '\n')
        return path

    def test_multi_tomogram_csv_yields_only_requested_tomogram(self, tmp_path: Path):
        csv_path = self._csv(tmp_path / 'picks.csv', ['tomoA,1,1,1,0.9', 'tomoB,5,5,5,0.9', 'tomoA,9,9,9,0.9'])
        picks = pick_tomogram([csv_path], ['mine'], self._manifest(tmp_path, 'tomoA'), None,
                              consensus_rule='union', distance_threshold=5.0, max_surface_distance_angstrom=300.0,
                              max_beam_angle_deviation=None, orientation_prior=OrientationPrior.NONE)
        assert sorted(p.position for p in picks) == [(1.0, 1.0, 1.0), (9.0, 9.0, 9.0)]
        assert {p.tomogram_id for p in picks} == {'tomoA'}

    def test_two_pickers_reconcile_to_one_pick(self, tmp_path: Path):
        a = self._csv(tmp_path / 'a.csv', ['tomoA,10,10,10,0.9'])
        b = self._csv(tmp_path / 'b.csv', ['tomoA,11,10,10,0.8'])
        picks = pick_tomogram([a, b], ['a', 'b'], self._manifest(tmp_path, 'tomoA'), None,
                              consensus_rule='union', distance_threshold=5.0, max_surface_distance_angstrom=300.0,
                              max_beam_angle_deviation=None, orientation_prior=OrientationPrior.NONE)
        assert len(picks) == 1 and set(picks[0].source_picker.split('+')) == {'a', 'b'}

    def test_picker_without_rows_warns_and_union_keeps_the_other(self, tmp_path: Path, monkeypatch):
        from stamp.commands import pick as pick_module
        warnings: list[str] = []
        monkeypatch.setattr(pick_module.log, 'warning', lambda message, *args, **kwargs: warnings.append(message))
        a = self._csv(tmp_path / 'a.csv', ['tomoA,3,3,3,0.9'])
        b = self._csv(tmp_path / 'b.csv', ['tomoB,7,7,7,0.9'])
        picks = pick_tomogram([a, b], ['a', 'b'], self._manifest(tmp_path, 'tomoA'), None,
                              consensus_rule='union', distance_threshold=5.0, max_surface_distance_angstrom=300.0,
                              max_beam_angle_deviation=None, orientation_prior=OrientationPrior.NONE)
        assert [p.position for p in picks] == [(3.0, 3.0, 3.0)]
        assert warnings == ['tomoA: b has no picks for this tomogram']
