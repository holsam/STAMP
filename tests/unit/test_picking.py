'''
STAMP: unit tests for picking functions, including geometric primitives, and consensus reconciliation
'''

# Import external dependencies
import mrcfile, numpy as np, pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.picking.consensus import build_particle_set, reconcile_picks
from stamp.utils.errors import StampValidationError
from stamp.picking.geometry import (
    beam_angle_deviation_degrees,
    compose_roll_about_normal,
    downsample_points,
    exclude_near_boundary,
    extract_surface,
    local_normalise,
    max_order_statistic_offset,
    non_maximum_suppression,
    quaternion_from_reference_to,
    robust_normalise,
    sample_along_normals,
    score_membrane_faces
)
from stamp.picking.vesicles import (
    load_vesicle_labels,
    summarise_vesicles,
    vesicle_ids_at,
    vesicle_surface_area_angstrom2,
)
from stamp.schemas.particles import HalfSet, Particle
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

    def test_downsample_reduces_points_and_is_deterministic(self) -> None:
        '''Downsampling shrinks the set reproducibly.'''
        vertices, normals = extract_surface(_hollow_sphere())
        first_points, first_normals = downsample_points(vertices, normals, spacing_voxels=4.0)
        second_points, _ = downsample_points(vertices, normals, spacing_voxels=4.0)
        assert first_points.shape[0] < vertices.shape[0]
        assert first_normals.shape[0] == first_points.shape[0]
        np.testing.assert_array_equal(first_points, second_points)

    def test_sample_along_normals_finds_a_planted_blob(self) -> None:
        '''The point under a planted blob scores highest'''
        # Flat membrane in the z=20 plane, with a bright blob sitting above it.
        volume = np.zeros((40, 40, 40), dtype=np.float32)
        volume[25, 20, 20] = 10.0

        points = np.array([[20.0, 20.0, 20.0], [20.0, 30.0, 30.0]])
        normals = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        densities = sample_along_normals(
            volume, points, normals,
            offset_min_voxels=4.0, offset_max_voxels=6.0, n_samples=3, direction=1,
        )
        # The point under the blob scores higher than the bare one.
        assert densities[0] > densities[1]

    def test_sample_along_normals_direction_matters(self) -> None:
        '''Sampling along the normal beats sampling against it'''
        volume = np.zeros((40, 40, 40), dtype=np.float32)
        volume[25, 20, 20] = 10.0
        points = np.array([[20.0, 20.0, 20.0]])
        normals = np.array([[1.0, 0.0, 0.0]])

        along = sample_along_normals(volume, points, normals, 4.0, 6.0, 3, direction=1)
        against = sample_along_normals(volume, points, normals, 4.0, 6.0, 3, direction=-1)
        assert along[0] > against[0]

    def test_robust_normalise_handles_flat_input(self) -> None:
        '''A flat input normalises to zeros'''
        normalised, _used_fallback = robust_normalise(np.ones(10))
        np.testing.assert_array_equal(normalised, np.zeros(10))

    def test_robust_normalise_scales_outlier(self) -> None:
        '''A lone outlier normalises well above threshold'''
        values = np.concatenate([np.zeros(99), [100.0]])
        normalised, _used_fallback = robust_normalise(values)
        assert normalised[-1] > 3.0

    def test_non_maximum_suppression_keeps_highest(self) -> None:
        '''NMS keeps the higher-scoring point in a close pair'''
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        scores = np.array([1.0, 5.0, 3.0])
        kept = non_maximum_suppression(points, scores, min_distance_voxels=10.0)
        assert set(kept.tolist()) == {1, 2}

    def test_non_maximum_suppression_empty_input(self) -> None:
        '''NMS on an empty input returns an empty result'''
        kept = non_maximum_suppression(np.empty((0, 3)), np.empty(0), 10.0)
        assert kept.size == 0

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

    def test_exclude_near_boundary(self) -> None:
        '''Points within the margin of a face are masked out'''
        points = np.array([[1.0, 1.0, 1.0], [20.0, 20.0, 20.0]])
        mask = exclude_near_boundary(points, (40, 40, 40), margin_voxels=5.0)
        assert mask.tolist() == [False, True]

    def test_faces_share_one_scale(self):
        '''A bright blob off one face does not inflate the other face's z-scores.'''
        tomo = np.zeros((40, 40, 40), dtype=np.float32)
        tomo[25, 20, 20] = -50.0  # dark blob on the +normal side of the y=20 vertex only
        ys = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
        vertices = np.stack([np.full_like(ys, 20.0), ys, np.full_like(ys, 20.0)], axis=1)
        normals = np.tile([1.0, 0.0, 0.0], (vertices.shape[0], 1))
        n_points = vertices.shape[0]
        contaminated = int(np.where(ys == 20.0)[0][0])
        _points, _n, scores, _w, _used_fallback, _mean, _profile = score_membrane_faces(tomo, vertices, normals, [(3.0, 6.0)], 3, -1)
        assert scores.shape == (2 * n_points,)
        pos_face, neg_face = scores[:n_points], scores[n_points:]
        clean = [i for i in range(n_points) if i != contaminated]
        # the blob sits only on the +normal face of the contaminated vertex
        assert abs(pos_face[contaminated]) > abs(neg_face[contaminated])
        # every clean vertex, on either face, stays near the untouched baseline
        for i in clean:
            assert abs(pos_face[i]) < 1e-6
            assert abs(neg_face[i]) < 1e-6

    def test_score_membrane_faces_reports_winning_window(self) -> None:
        '''A point with density only in the second window is won by that window, not the first.'''
        tomo = np.zeros((60, 60, 60), dtype=np.float32)
        tomo[20, 20, 55] = -5.0  # offset 35 voxels along +x from the point below
        vertices = np.array([[20.0, 20.0, 20.0]])
        normals = np.array([[0.0, 0.0, 1.0]])
        windows = [(2.0, 10.0), (30.0, 40.0)]
        _points, _n, scores, winning_window, _used_fallback, _mean, _profile = score_membrane_faces(tomo, vertices, normals, windows, 5, -1)
        # two faces per point; the +x face (index 0) sees the density, the -x face (index 1) sees nothing
        assert winning_window[0] == 1
        assert scores[0] > scores[1]

    def test_max_order_statistic_offset_zero_for_single_window(self) -> None:
        assert max_order_statistic_offset(1) == 0.0

    def test_max_order_statistic_offset_increases_with_window_count(self) -> None:
        assert 0.0 < max_order_statistic_offset(2) < max_order_statistic_offset(4) < max_order_statistic_offset(8)

    def test_max_order_statistic_offset_rejects_zero_windows(self) -> None:
        with pytest.raises(ValueError, match='n_windows'):
            max_order_statistic_offset(0)

    def test_score_membrane_faces_rejects_no_windows(self) -> None:
        with pytest.raises(ValueError, match='offset_windows_voxels'):
            score_membrane_faces(np.zeros((10, 10, 10)), np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), [], 5, -1)

    # test_compose_roll_is_identity_at_zero: zero angle leaves the normal quaternion unchanged
    def test_compose_roll_is_identity_at_zero(self):
        base = quaternion_from_reference_to(np.array([0.3, -0.5, 0.8]))
        rolled = compose_roll_about_normal(base, 0.0)
        assert np.allclose(rolled, base, atol=1e-9)

    # test_compose_roll_preserves_normal: rolling about the normal does not move +z's image
    def test_compose_roll_preserves_normal(self):
        from stamp.classify.extract import quaternion_to_matrix
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
            Particle(particle_id='p0', tomogram_id='t', position=(0, 0, 0), source_picker='x', half_set=HalfSet.A, beam_angle_deviation_degrees=5.0),
            Particle(particle_id='p1', tomogram_id='t', position=(0, 0, 0), source_picker='x', half_set=HalfSet.A, beam_angle_deviation_degrees=85.0),
        ]
        kept = [p for p in particles if p.beam_angle_deviation_degrees is None or p.beam_angle_deviation_degrees <= 45.0]
        assert [p.particle_id for p in kept] == ['p0']

    def test_filter_retains_all_when_disabled(self) -> None:
        particles = [
            Particle(particle_id='p0', tomogram_id='t', position=(0, 0, 0), source_picker='x', half_set=HalfSet.A, beam_angle_deviation_degrees=5.0),
            Particle(particle_id='p1', tomogram_id='t', position=(0, 0, 0), source_picker='x', half_set=HalfSet.A, beam_angle_deviation_degrees=85.0),
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
