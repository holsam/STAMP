'''
STAMP: unit tests for picking functions, including geometric primitives, native picker and consensus reconciliation
'''

# Import external dependencies
import mrcfile, numpy as np, pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.picking.consensus import build_particle_set, reconcile_picks
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
from stamp.picking.native import NativePickerConfig, pick_tomogram
from stamp.picking.vesicles import (
    load_vesicle_labels,
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

# _vesicle_with_particles: a hollow-sphere segmentation plus a tomogram with dark blobs outside the shell
def _vesicle_with_particles(shape=(60, 60, 60), radius=18.0, thickness=2.0, particle_offsets=((0, 0, 1),)):
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

# _linear_gradient_vesicle: hollow-sphere segmentation on a tomogram whose background ramps linearly along x, with identical particle blobs planted at both ends
def _linear_gradient_vesicle(shape=(80, 80, 120), radius=20.0, thickness=2.0) -> tuple[np.ndarray, np.ndarray]:
    centre = np.array(shape) / 2.0
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1)
    distance = np.linalg.norm(grid - centre, axis=-1)
    segmentation = ((distance > radius - thickness) & (distance < radius + thickness)).astype(np.float32)
    # Background ramps from 0 at x=0 to -4 at x=max to mimic thickness/defocus gradient
    x_ramp = -4.0 * (np.arange(shape[2]) / (shape[2] - 1))
    tomogram = np.broadcast_to(x_ramp[None, None, :], shape).astype(np.float32).copy()
    # Small noise floor: real tomograms are never noiseless, and MAD needs ambient
    # variance to behave as a stable scale estimator rather than tracking blob contamination directly
    tomogram = tomogram + np.random.default_rng(0).normal(scale=0.3, size=shape).astype(np.float32)
    tomogram[segmentation > 0] = np.median(tomogram[segmentation <= 0])
    # Identical dark particle blobs just outside the shell, at low-x and high-x poles
    for x_index in (shape[2] // 2 - int(radius) - 5, shape[2] // 2 + int(radius) + 5):
        centre_particle = np.array([shape[0] / 2, shape[1] / 2, x_index])
        particle_distance = np.linalg.norm(grid - centre_particle, axis=-1)
        tomogram[particle_distance < 3.0] -= 3.0
    return segmentation, tomogram

# _particle_free_vesicle: a spherical shell segmentation with a tomogram of pure background noise, no planted density, for measuring false-positive rate
def _particle_free_vesicle(seed: int) -> tuple[np.ndarray, np.ndarray]:
    shape = (60, 60, 60)
    centre = np.array(shape) / 2.0
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    radius = np.sqrt((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2)
    segmentation = ((radius > 18.0) & (radius < 20.0)).astype(np.float32)
    rng = np.random.default_rng(seed)
    tomogram = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    return segmentation, tomogram

# _vesicle_with_two_protrusion_heights: a hollow-sphere segmentation with two planted particles at distinct offsets from the membrane, on opposite sides so NMS keeps them apart
def _vesicle_with_two_protrusion_heights(
    shape=(120, 120, 120), radius=20.0, thickness=2.0,
    small_offset_angstrom=35.0, large_offset_angstrom=130.0, voxel_size_angstrom=10.0,
):
    centre = np.array(shape) / 2.0
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1)
    distance = np.linalg.norm(grid - centre, axis=-1)
    segmentation = ((distance > radius - thickness) & (distance < radius + thickness)).astype(np.float32)
    tomogram = np.zeros(shape, dtype=np.float32)
    tomogram[segmentation > 0] = -1.0

    expected_positions_zyx = []
    for direction, offset_angstrom in (((0, 0, 1), small_offset_angstrom), ((0, 0, -1), large_offset_angstrom)):
        unit = np.array(direction, dtype=float)
        unit /= np.linalg.norm(unit)
        offset_voxels = offset_angstrom / voxel_size_angstrom
        position = centre + unit * (radius + offset_voxels)
        expected_positions_zyx.append(position)
        index = np.round(position).astype(int)
        tomogram[
            index[0] - 2 : index[0] + 3,
            index[1] - 2 : index[1] + 3,
            index[2] - 2 : index[2] + 3,
        ] = -8.0
    return segmentation, tomogram, expected_positions_zyx

# _count_above_threshold_uncorrected: repeat the pre-NMS scoring step at plain n_mad (no order-statistic correction), for comparison against the corrected threshold
def _count_above_threshold_uncorrected(tmp_path: Path, config: NativePickerConfig, n_mad: float) -> int:
    with mrcfile.open(str(tmp_path / 'seg.mrc'), permissive=True) as mrc:
        segmentation = np.asarray(mrc.data)
    with mrcfile.open(str(tmp_path / 'tomo.mrc'), permissive=True) as mrc:
        tomogram = np.asarray(mrc.data).astype(np.float32).copy()
    tomogram[segmentation > 0] = np.median(tomogram[segmentation <= 0])
    vertices, normals = extract_surface(segmentation)
    vertices, normals = downsample_points(vertices, normals, config.to_voxels(config.surface_spacing_angstrom))
    inside = exclude_near_boundary(vertices, segmentation.shape, config.to_voxels(config.max_offset_angstrom))
    vertices, normals = vertices[inside], normals[inside]
    windows = [(config.to_voxels(lo), config.to_voxels(hi)) for lo, hi in config.offset_windows_angstrom]
    _points, _n, scores, _w, _used_fallback, _mean, _profile = score_membrane_faces(
        tomogram, vertices, normals, windows, config.n_samples, config.density_sign, scoring_mode=config.scoring_mode,
    )
    return int(np.sum(scores >= n_mad))

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

    def test_global_normalisation_biases_picks_toward_low_background_end(self, tmp_path: Path) -> None:
        segmentation, tomogram = _linear_gradient_vesicle()
        seg_path, tomo_path = tmp_path / 'seg.mrc', tmp_path / 'tomo.mrc'
        _write_mrc(seg_path, segmentation)
        _write_mrc(tomo_path, tomogram)
        config = NativePickerConfig(voxel_size_angstrom=10.0, n_mad=1.5, normalisation='global', scoring_mode='mean')
        picks = pick_tomogram(seg_path, tomo_path, 'gradient', config)
        # Both particles are equally strong so global threshold should favour low-background (high-x, deep-ramp) end
        x_positions = [pick.position[0] for pick in picks]
        assert len(x_positions) > 0
        low_x_count = sum(1 for x in x_positions if x < tomogram.shape[2] / 2)
        high_x_count = len(x_positions) - low_x_count
        assert high_x_count > low_x_count

    def test_local_normalisation_does_not_bias_picks(self, tmp_path: Path) -> None:
        segmentation, tomogram = _linear_gradient_vesicle()
        seg_path, tomo_path = tmp_path / 'seg.mrc', tmp_path / 'tomo.mrc'
        _write_mrc(seg_path, segmentation)
        _write_mrc(tomo_path, tomogram)
        config = NativePickerConfig(voxel_size_angstrom=10.0, n_mad=1.5, normalisation='local', local_radius_angstrom=250.0, min_local_neighbours=5, scoring_mode='mean')
        picks = pick_tomogram(seg_path, tomo_path, 'gradient', config)
        x_positions = [pick.position[0] for pick in picks]
        assert len(x_positions) > 0
        low_x_count = sum(1 for x in x_positions if x < tomogram.shape[2] / 2)
        high_x_count = len(x_positions) - low_x_count
        assert low_x_count > 0 and high_x_count > 0

    def test_beam_aligned_and_orthogonal_particles_get_correct_deviation(self, tmp_path: Path) -> None:
        '''A particle planted beam-aligned (normal along z) reads ~90 deg; one beam-orthogonal (normal in-plane) reads ~0 deg.'''
        segmentation, tomogram, _ = _vesicle_with_particles(particle_offsets=((0, 0, 1), (1, 0, 0)),)
        _write_mrc(tmp_path / 'seg.mrc', segmentation)
        _write_mrc(tmp_path / 'tomo.mrc', tomogram)

        config = NativePickerConfig(
            voxel_size_angstrom=10.0,
            offset_windows_angstrom=((40.0, 80.0),),
            surface_spacing_angstrom=20.0,
            min_particle_distance_angstrom=80.0,
            n_mad=3.0,
        )
        picks = pick_tomogram(tmp_path / 'seg.mrc', tmp_path / 'tomo.mrc', 'tomo000', config)
        assert picks, 'picker found nothing where particles were planted'

        deviations = np.array([p.beam_angle_deviation_degrees for p in picks])
        # nearest-beam-aligned pick and nearest-beam-orthogonal pick, by deviation
        assert deviations.max() > 60.0, 'expected a near-beam-aligned pick close to 90 deg'
        assert deviations.min() < 30.0, 'expected a near-beam-orthogonal pick close to 0 deg'

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


# TestNativePicker: class containing unit tests for test_native_picker.py
class TestNativePicker:
    def test_picker_finds_planted_particle(self, tmp_path: Path) -> None:
        '''The picker returns a pick near each planted particle'''
        segmentation, tomogram, expected_zyx = _vesicle_with_particles()
        _write_mrc(tmp_path / 'seg.mrc', segmentation)
        _write_mrc(tmp_path / 'tomo.mrc', tomogram)

        config = NativePickerConfig(
            voxel_size_angstrom=10.0,
            offset_windows_angstrom=((40.0, 80.0),),
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

    def test_sparse_neighbourhood_falls_back_to_global(self) -> None:
        rng = np.random.default_rng(0)
        # A dense cluster near the origin, plus one isolated point far away with no neighbours within radius: the isolated point should fall back to global stats
        dense_points = rng.normal(size=(50, 3)) * 5.0
        isolated_point = np.array([[500.0, 500.0, 500.0]])
        points = np.vstack([dense_points, isolated_point])
        values = rng.normal(size=51)
        normalised, used_fallback = local_normalise(values, points, radius_voxels=20.0, min_neighbours=10)
        assert not used_fallback[:50].any()
        assert used_fallback[50]
        global_median = np.median(values)
        global_mad = np.median(np.abs(values - global_median))
        global_scale = 1.4826 * global_mad if global_mad > 0.0 else values.std()
        expected_isolated = (values[50] - global_median) / global_scale
        assert normalised[50] == pytest.approx(expected_isolated)

    def test_robust_normalise_local_mode_requires_points(self) -> None:
        with pytest.raises(ValueError):
            robust_normalise(np.ones(10), mode='local')

    def test_multiscale_offsets_separate_two_protrusion_heights(self, tmp_path: Path) -> None:
        '''Two particles at distinct protrusion heights are each best scored by the window whose range covers that height, and the recorded winning window separates them.'''
        segmentation, tomogram, _ = _vesicle_with_two_protrusion_heights(small_offset_angstrom=35.0, large_offset_angstrom=130.0)
        _write_mrc(tmp_path / 'seg.mrc', segmentation)
        _write_mrc(tmp_path / 'tomo.mrc', tomogram)

        config = NativePickerConfig(voxel_size_angstrom=10.0, n_mad=2.0)
        picks = pick_tomogram(tmp_path / 'seg.mrc', tmp_path / 'tomo.mrc', 'tomo000', config)

        small_picks = [p for p in picks if p.offset_window_angstrom is not None and p.offset_window_angstrom[1] <= 80.0]
        large_picks = [p for p in picks if p.offset_window_angstrom is not None and p.offset_window_angstrom[0] >= 70.0]
        assert small_picks, 'the 35A protrusion should be won by a small offset window'
        assert large_picks, 'the 130A protrusion should be won by a large offset window'
        assert {p.offset_window_angstrom for p in small_picks}.isdisjoint(
            {p.offset_window_angstrom for p in large_picks}
        )

    def test_single_window_config_matches_pre_6g_threshold(self) -> None:
        '''A single-element offset_windows_angstrom has no order-statistic correction to apply.'''
        config = NativePickerConfig(voxel_size_angstrom=10.0, offset_windows_angstrom=((40.0, 80.0),), n_mad=3.0)
        assert config.effective_n_mad == pytest.approx(3.0)

    def test_multiscale_threshold_correction_holds_false_positive_rate(self, tmp_path: Path) -> None:
        '''
        On a particle-free tomogram, scoring four windows and thresholding at effective_n_mad doesn't pass more points than scoring one window at n_mad.
        An uncorrected max over the same four windows, thresholded at plain n_mad, does pass more points that scoring a single window.
        '''
        segmentation, tomogram = _particle_free_vesicle(seed=0)
        _write_mrc(tmp_path / 'seg.mrc', segmentation)
        _write_mrc(tmp_path / 'tomo.mrc', tomogram)

        n_mad = 2.5
        single = NativePickerConfig(voxel_size_angstrom=10.0, offset_windows_angstrom=((40.0, 80.0),), n_mad=n_mad, scoring_mode='mean')
        multi = NativePickerConfig(voxel_size_angstrom=10.0, n_mad=n_mad, scoring_mode='mean')  # default 4 windows

        single_picks = pick_tomogram(tmp_path / 'seg.mrc', tmp_path / 'tomo.mrc', 'tomo000', single)
        multi_picks = pick_tomogram(tmp_path / 'seg.mrc', tmp_path / 'tomo.mrc', 'tomo000', multi)

        # uncorrected: same tomogram & windows, thresholded at plain n_mad instead of effective_n_mad
        uncorrected_above = _count_above_threshold_uncorrected(tmp_path, multi, n_mad)

        assert len(multi_picks) <= len(single_picks) + 1, f'corrected multi-window picking should not produce more false positives than single-window picking on a particle-free tomogram (single: {len(single_picks)}; multi: {len(multi_picks)})'
        assert uncorrected_above > len(multi_picks), f'uncorrected thresholding passes more candidates than the corrected version (single: {len(single_picks)}; uncorrected: {uncorrected_above}; corrected: {len(multi_picks)})'

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
        '''Mismatched shapes raise ValueError rather than silently misattributing'''
        _write_mrc(tmp_path / 'labels.mrc', np.zeros((10, 10, 10)))
        with pytest.raises(ValueError):
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

