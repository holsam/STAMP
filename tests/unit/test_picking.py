'''
STAMP: unit tests for picking functions, including geometric primitives, native picker and consensus reconciliation
'''

# Import external dependencies
import mrcfile, numpy as np, pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.picking.consensus import build_particle_set, reconcile_picks
from stamp.picking.geometry import (
    downsample_points,
    exclude_near_boundary,
    extract_surface,
    non_maximum_suppression,
    quaternion_from_reference_to,
    robust_normalise,
    sample_along_normals,
    score_membrane_faces
)
from stamp.picking.native import NativePickerConfig, pick_tomogram
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

# _vesicle_with_particles: a hollow-sphere segmentation plus a tomogram with dark blobs outside the shell
def _vesicle_with_particles(
    shape=(60, 60, 60), radius=18.0, thickness=2.0, particle_offsets=((0, 0, 1),)
):
    '''A hollow sphere segmentation plus a matching tomogram with dark blobs planted just outside the shell along the given directions'''
    centre = np.array(shape) / 2.0
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1)
    distance = np.linalg.norm(grid - centre, axis=-1)
    segmentation = (
        (distance > radius - thickness) & (distance < radius + thickness)
    ).astype(np.float32)
    # Membrane itself is dark; particles are darker still, sitting ~6 voxels out
    tomogram = np.zeros(shape, dtype=np.float32)
    tomogram[segmentation > 0] = -1.0
    expected_positions_zyx = []
    for direction in particle_offsets:
        unit = np.array(direction, dtype=float)
        unit /= np.linalg.norm(unit)
        position = centre + unit * (radius + 6.0)
        expected_positions_zyx.append(position)
        index = np.round(position).astype(int)
        # A small dark blob
        tomogram[
            index[0] - 2 : index[0] + 3,
            index[1] - 2 : index[1] + 3,
            index[2] - 2 : index[2] + 3,
        ] = -8.0
    return segmentation, tomogram, expected_positions_zyx

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
        with pytest.raises(ValueError, match='no voxels above'):
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
        np.testing.assert_array_equal(
            robust_normalise(np.ones(10)), np.zeros(10)
        )

    def test_robust_normalise_scales_outlier(self) -> None:
        '''A lone outlier normalises well above threshold'''
        values = np.concatenate([np.zeros(99), [100.0]])
        normalised = robust_normalise(values)
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
        _points, _n, scores = score_membrane_faces(tomo, vertices, normals, 3.0, 6.0, 3, -1)
        assert scores.shape == (2 * n_points,)
        pos_face, neg_face = scores[:n_points], scores[n_points:]
        clean = [i for i in range(n_points) if i != contaminated]
        # the blob sits only on the +normal face of the contaminated vertex
        assert abs(pos_face[contaminated]) > abs(neg_face[contaminated])
        # every clean vertex, on either face, stays near the untouched baseline
        for i in clean:
            assert abs(pos_face[i]) < 1e-6
            assert abs(neg_face[i]) < 1e-6

# TestNativePicker: class containing unit tests for test_native_picker.py
class TestNativePicker:
    def test_picker_finds_planted_particle(self, tmp_path: Path) -> None:
        '''The picker returns a pick near each planted particle'''
        segmentation, tomogram, expected_zyx = _vesicle_with_particles()
        _write_mrc(tmp_path / 'seg.mrc', segmentation)
        _write_mrc(tmp_path / 'tomo.mrc', tomogram)

        config = NativePickerConfig(
            voxel_size_angstrom=10.0,
            offset_min_angstrom=40.0,
            offset_max_angstrom=80.0,
            surface_spacing_angstrom=20.0,
            min_particle_distance_angstrom=80.0,
            n_mad=3.0,
            density_sign=-1,
        )
        picks = pick_tomogram(tmp_path / 'seg.mrc', tmp_path / 'tomo.mrc', 'tomo000', config)

        assert picks, 'picker found nothing where a particle was planted'

        # Pick positions are (x, y, z); expected are (z, y, x).
        expected_xyz = np.array([position[::-1] for position in expected_zyx])
        found_xyz = np.array([pick.position for pick in picks])
        distances = np.linalg.norm(found_xyz[:, None, :] - expected_xyz[None, :, :], axis=-1)
        assert distances.min() < 10.0, (
            f'nearest pick was {distances.min():.1f} voxels from the planted particle'
        )

    def test_picker_returns_nothing_on_featureless_membrane(self, tmp_path: Path) -> None:
        '''A vesicle with no particles should yield few or no picks (robust normalisation means some points always sit above the median, so this asserts a low count rather than exactly zero)'''
        segmentation, tomogram, _ = _vesicle_with_particles(particle_offsets=())
        _write_mrc(tmp_path / 'seg.mrc', segmentation)
        _write_mrc(tmp_path / 'tomo.mrc', tomogram)

        config = NativePickerConfig(voxel_size_angstrom=10.0, n_mad=4.0)
        picks = pick_tomogram(tmp_path / 'seg.mrc', tmp_path / 'tomo.mrc', 'tomo000', config)
        assert len(picks) <= 3, f'expected few picks on a featureless membrane, got {len(picks)}'

    def test_picker_output_conventions(self, tmp_path: Path) -> None:
        '''Every pick carries the expected id, picker, orientation and bounds'''
        segmentation, tomogram, _ = _vesicle_with_particles()
        _write_mrc(tmp_path / 'seg.mrc', segmentation)
        _write_mrc(tmp_path / 'tomo.mrc', tomogram)

        config = NativePickerConfig(voxel_size_angstrom=10.0, n_mad=2.0)
        picks = pick_tomogram(tmp_path / 'seg.mrc', tmp_path / 'tomo.mrc', 'tomo000', config)

        for pick in picks:
            assert pick.tomogram_id == 'tomo000'
            assert pick.source_picker == 'stamp-native'
            assert pick.orientation is not None  # unit-length enforced by the schema
            assert pick.confidence is not None
            for coordinate, extent in zip(pick.position, (60, 60, 60)):
                assert 0 <= coordinate <= extent

    def test_picker_rejects_mismatched_shapes(self, tmp_path: Path) -> None:
        '''Mismatched segmentation/tomogram shapes raise a clear error'''
        _write_mrc(tmp_path / 'seg.mrc', np.ones((20, 20, 20), dtype=np.float32))
        _write_mrc(tmp_path / 'tomo.mrc', np.zeros((30, 30, 30), dtype=np.float32))
        config = NativePickerConfig(voxel_size_angstrom=10.0)
        with pytest.raises(ValueError, match='does not match tomogram shape'):
            pick_tomogram(tmp_path / 'seg.mrc', tmp_path / 'tomo.mrc', 'tomo000', config)

    def test_config_rejects_bad_density_sign(self) -> None:
        '''A density_sign other than +1 or -1 raises'''
        with pytest.raises(ValueError, match='density_sign'):
            NativePickerConfig(voxel_size_angstrom=10.0, density_sign=0)

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
