'''
STAMP: unit tests for refinement FSC and halfset guard functions
'''

# Import external dependencies
import numpy as np, pytest

# Import internal STAMP objects
from stamp.refine.fsc import compute_fsc, soft_sphere_mask
from stamp.refine.halfset_guard import (
    assert_distinct_references, refine_output_tree, split_class_by_half,
)
from stamp.schemas.particles import ClassAssignment, HalfSet, Particle

# _particle: construct a Particle instance for tests
def _particle(pid, half):
    return Particle(
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
        with pytest.raises(ValueError, match='both halves'):
            split_class_by_half('c00', assignments, particles)

    def test_assert_distinct_references_rejects_shared_seed(self, tmp_path):
        reference = tmp_path / 'seed.mrc'
        reference.write_bytes(b'0')
        with pytest.raises(ValueError, match='each half must start from its own class average'):
            assert_distinct_references(reference, reference)

    def test_output_tree_is_separate_per_half(self, tmp_path):
        tree = refine_output_tree(tmp_path, 'c00')
        assert tree['A'] != tree['B'] and tree['A'].is_dir() and tree['B'].is_dir()
