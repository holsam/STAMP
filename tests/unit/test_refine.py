'''
STAMP: unit tests for refinement FSC and halfset guard functions
'''

# Import external dependencies
import numpy as np, pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.refine.fsc import compute_fsc, soft_sphere_mask, _shell_fsc, _shell_map
from stamp.refine.halfset_guard import (
    assert_distinct_references, assert_independent_outputs, refine_output_tree, split_class_by_half,
)
from stamp.schemas.particles import ClassAssignment, HalfSet, OrientationPrior, Particle
from stamp.utils.errors import StampValidationError

# _particle: construct a Particle instance for tests
def _particle(pid, half):
    return Particle(
        prior=OrientationPrior.NORMAL,
        particle_id=pid,
        tomogram_id='t0',
        position=(1.0, 1.0, 1.0),
        source_picker='x',
        half_set=half
    )

# TestGeometry: unit tests for refine/fsc.py
class TestRefineFsc:
    def test_fsc_of_identical_maps_is_one(self):
        rng = np.random.default_rng(0)
        volume = rng.random((24, 24, 24))
        result = compute_fsc(volume, volume.copy(), voxel_size_angstrom=3.0)
        assert np.allclose(result.fsc_masked[result.fsc_masked != 0.0], 1.0, atol=1e-6)

    def test_fsc_of_independent_noise_is_near_zero(self):
        rng = np.random.default_rng(1)
        a = rng.random((24, 24, 24))
        b = rng.random((24, 24, 24))
        result = compute_fsc(a, b, voxel_size_angstrom=3.0)
        assert np.abs(result.fsc_masked[2:]).mean() < 0.3

    def test_resolution_readout_picks_first_subthreshold_shell(self):
        rng = np.random.default_rng(2)
        base = rng.random((24, 24, 24))
        noisy = base + 0.01 * rng.random((24, 24, 24))
        result = compute_fsc(base, noisy, voxel_size_angstrom=2.0)
        crossing = np.argwhere(result.fsc_masked < 0.143).ravel()
        if crossing.size and crossing[0] > 0:
            expected = 1.0 / result.frequencies_per_angstrom[crossing[0]]
            assert result.resolution_angstrom == pytest.approx(expected)

    def test_mask_changes_the_curve(self):
        rng = np.random.default_rng(3)
        a = rng.random((24, 24, 24))
        b = a + 0.2 * rng.random((24, 24, 24))
        masked = compute_fsc(a, b, 3.0, mask=soft_sphere_mask(a.shape, radius_fraction=0.4))
        default = compute_fsc(a, b, 3.0)
        assert not np.allclose(masked.fsc_masked, default.fsc_masked)

    def test_identical_half_maps_correlate_at_dc(self):
        '''Shell 0 is ~1 for identical maps.'''
        rng = np.random.default_rng(0)
        volume = rng.random((24, 24, 24)).astype(np.float32)
        result = compute_fsc(volume, volume.copy(), voxel_size_angstrom=4.0)
        assert result.fsc_masked[0] > 0.99

    def test_uncorrelated_maps_drop_off(self):
        '''Independent noise gives a low high-frequency FSC.'''
        rng = np.random.default_rng(1)
        a = rng.random((24, 24, 24)).astype(np.float32)
        b = rng.random((24, 24, 24)).astype(np.float32)
        result = compute_fsc(a, b, voxel_size_angstrom=4.0)
        assert result.fsc_masked[-1] < 0.3

    def test_correction_lowers_masked_fsc(self):
        '''Correlated noise under a mask is pulled down.'''
        rng = np.random.default_rng(2)
        signal = rng.random((28, 28, 28)).astype(np.float32)
        a = signal + rng.normal(0, 1.0, signal.shape).astype(np.float32)
        b = signal + rng.normal(0, 1.0, signal.shape).astype(np.float32)
        mask = soft_sphere_mask(a.shape, radius_fraction=0.5)
        raw = _shell_fsc(a * mask, b * mask, *_shell_map(a.shape))
        corrected = compute_fsc(a, b, 4.0, mask=mask).fsc_masked
        assert corrected[-1] <= raw[-1] + 1e-6

# TestGeometry: unit tests for refine/halfset_guard.py
class TestRefineHalfsetGuard:
    def test_split_class_by_half(self):
        particles = [_particle('p0', HalfSet.A), _particle('p1', HalfSet.B), _particle('p2', HalfSet.A)]
        assignments = [ClassAssignment(particle_id=p.particle_id, cluster_id='c00', classifier='k') for p in particles]
        half_a, half_b = split_class_by_half('c00', assignments, particles)
        assert {p.particle_id for p in half_a} == {'p0', 'p2'} and [p.particle_id for p in half_b] == ['p1']

    def test_split_raises_on_empty_half(self):
        particles = [_particle('p0', HalfSet.A)]
        assignments = [ClassAssignment(particle_id='p0', cluster_id='c00', classifier='k')]
        with pytest.raises(StampValidationError, match='both halves'):
            split_class_by_half('c00', assignments, particles)

    def test_assert_distinct_references_rejects_shared_seed(self, tmp_path):
        reference = tmp_path / 'seed.mrc'
        reference.write_bytes(b'0')
        with pytest.raises(StampValidationError, match='each half must start from its own class average'):
            assert_distinct_references(reference, reference)

    def test_output_tree_is_separate_per_half(self, tmp_path):
        tree = refine_output_tree(tmp_path, 'c00')
        assert tree['A'] != tree['B'] and tree['A'].is_dir() and tree['B'].is_dir()

    def test_assert_independent_outputs_rejects_same_file(self, tmp_path):
        path = tmp_path / 'a' / 'final.mrc'
        path.parent.mkdir()
        m, r = np.ones((4, 4, 4)), np.zeros((4, 4, 4))
        with pytest.raises(StampValidationError, match='each half must be its own refinement output'):
            assert_independent_outputs(path, path, m, m + 1, r, r)

    def test_assert_independent_outputs_rejects_same_directory(self, tmp_path):
        m, r = np.ones((4, 4, 4)), np.zeros((4, 4, 4))
        with pytest.raises(StampValidationError, match='share a directory'):
            assert_independent_outputs(tmp_path / 'x.mrc', tmp_path / 'y.mrc', m, m + 1, r, r)

    def test_assert_independent_outputs_rejects_differing_shapes(self, tmp_path):
        (tmp_path / 'a').mkdir(); (tmp_path / 'b').mkdir()
        with pytest.raises(StampValidationError, match='boxes differ'):
            assert_independent_outputs(tmp_path / 'a' / 'x.mrc', tmp_path / 'b' / 'y.mrc', np.ones((4, 4, 4)), np.ones((6, 6, 6)), np.zeros((4, 4, 4)), np.zeros((6, 6, 6)))

    def test_assert_independent_outputs_rejects_unrefined_seed(self, tmp_path):
        (tmp_path / 'a').mkdir(); (tmp_path / 'b').mkdir()
        seed = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        with pytest.raises(StampValidationError, match='identical to its seed'):
            assert_independent_outputs(tmp_path / 'a' / 'x.mrc', tmp_path / 'b' / 'y.mrc', seed, seed + 1, seed, np.zeros((4, 4, 4), np.float32))

    def test_assert_independent_outputs_rejects_identical_maps(self, tmp_path):
        (tmp_path / 'a').mkdir(); (tmp_path / 'b').mkdir()
        m = np.ones((4, 4, 4), np.float32)
        with pytest.raises(StampValidationError, match='identical'):
            assert_independent_outputs(tmp_path / 'a' / 'x.mrc', tmp_path / 'b' / 'y.mrc', m, m.copy(), np.zeros((4, 4, 4), np.float32), np.zeros((4, 4, 4), np.float32))

    def test_assert_independent_outputs_accepts_distinct_refinements(self, tmp_path):
        (tmp_path / 'a').mkdir(); (tmp_path / 'b').mkdir()
        rng = np.random.default_rng(0)
        a, b = rng.random((4, 4, 4)), rng.random((4, 4, 4))
        assert_independent_outputs(tmp_path / 'a' / 'x.mrc', tmp_path / 'b' / 'y.mrc', a, b, np.zeros((4, 4, 4)), np.ones((4, 4, 4)))

# TestRefineCommand: class containing tests for src/stamp/commands/refine.py
class TestRefineCommand:
    # test_full_poses_come_from_assignments_the_polish_step_posed: only rows with a rotation produce a pose, a missing shift is zero
    def test_full_poses_come_from_assignments_the_polish_step_posed(self) -> None:
        from stamp.refine.halfset_guard import full_poses
        from stamp.schemas.particles import ClassAssignment
        rows = [
            ClassAssignment(particle_id='a', cluster_id='c00', classifier='x', rotation=(1.0, 0.0, 0.0, 0.0), shift_voxels=(1.0, 2.0, 3.0)),
            ClassAssignment(particle_id='b', cluster_id='c00', classifier='x', rotation=(1.0, 0.0, 0.0, 0.0)),
            ClassAssignment(particle_id='c', cluster_id='c00', classifier='x', inplane_angle_degrees=10.0),
        ]
        poses = full_poses(rows)
        assert set(poses) == {'a', 'b'} and list(poses['a'][1]) == [1.0, 2.0, 3.0] and not poses['b'][1].any()

# _relion_matrix: RELION 5 Euler::anglesToMatrix3 (jaz/math/Euler_angles_relion.h), the matrix mapping particle space to tomogram space
def _relion_matrix(rot: float, tilt: float, psi: float):
    import numpy as np
    p, t, c = np.radians([rot, tilt, psi])
    sp, cp, st, ct, sc, cc = np.sin(p), np.cos(p), np.sin(t), np.cos(t), np.sin(c), np.cos(c)
    return np.array([[cc * ct * cp - sc * sp, cc * ct * sp + sc * cp, -cc * st], [-sc * ct * cp - cc * sp, -sc * ct * sp + cc * cp, sc * st], [st * cp, st * sp, ct]])

# TestRelionPoses: class containing tests for writing class poses into RELION particle files
class TestRelionPoses:
    def test_euler_angles_round_trip(self) -> None:
        import numpy as np
        from scipy.spatial.transform import Rotation
        from stamp.refine.relion_export import relion_euler_angles
        for seed in range(5):
            rotation = Rotation.random(random_state=seed).as_matrix()
            rot, tilt, psi = relion_euler_angles(rotation)
            assert np.allclose(_relion_matrix(rot, tilt, psi), rotation)
            assert 0.0 <= tilt <= 180.0

    def test_pose_is_composed_with_the_box_orientation_and_origin_is_in_angstrom(self) -> None:
        import numpy as np
        from scipy.spatial.transform import Rotation
        from stamp.refine.relion_export import pose_to_relion
        from stamp.classify.frames import azimuth_matrix, rotation_to_quaternion
        orientation = rotation_to_quaternion(Rotation.from_euler('y', 90, degrees=True).as_matrix())
        row = pose_to_relion(orientation, (azimuth_matrix(30.0), np.array([2.0, 0.0, 0.0])), 5.0)
        composed = Rotation.from_euler('y', 90, degrees=True).as_matrix() @ azimuth_matrix(30.0)
        assert np.allclose(_relion_matrix(row['rlnAngleRot'], row['rlnAngleTilt'], row['rlnAnglePsi']), composed)
        # a 2 voxel box x shift is -z in the tomogram under this orientation, the origin is the negative of it, 5 A per voxel
        assert np.allclose([row['rlnOriginXAngst'], row['rlnOriginYAngst'], row['rlnOriginZAngst']], [0.0, 0.0, 10.0], atol=1e-6)

    def test_star_carries_poses_when_given_and_priors_otherwise(self, tmp_path: Path) -> None:
        import numpy as np, starfile
        from stamp.refine.relion_export import write_particle_star
        from stamp.schemas.particles import HalfSet, Particle
        make = lambda i: Particle(prior=OrientationPrior.NORMAL, particle_id=f'p{i}', tomogram_id='t', position=(1.0, 2.0, 3.0), orientation=(1.0, 0.0, 0.0, 0.0), source_picker='x', half_set=HalfSet.A)
        write_particle_star([make(0), make(1)], tmp_path / 'p.star', poses={'p0': (np.eye(3), np.zeros(3))}, voxel_size_angstrom=4.0)
        table = starfile.read(tmp_path / 'p.star').set_index('rlnTomoParticleName')
        assert 'rlnOriginXAngst' in table.columns and table.loc['p0', 'rlnOriginXAngst'] == 0.0
        assert table.loc['p1', 'rlnAngleTiltPrior'] == 0.0     # no pose for p1, so the old normal-derived prior path
