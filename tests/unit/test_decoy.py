'''
STAMP: unit tests for decoy generation and the decoy/real separation guards
'''

# Import external dependencies
import mrcfile, numpy as np, pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.decoy.generate import DecoyGeometry, generate_shifted_decoys, generate_synthetic_noise_decoys
from stamp.decoy.validate import (
    assert_comparable,
    check_manifest_purity,
    is_decoy_particle_set,
)
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.particles import HalfSet, Particle, ParticleSet
from stamp.utils.errors import StampPipelineError, StampValidationError

# _write_vesicle_pair: write a matched segmentation/tomogram vesicle pair and return its manifest
def _write_vesicle_pair(tmp_path: Path, tomogram_id: str = 'tomo000') -> TomogramManifest:
    shape = (60, 60, 60)
    centre = np.array(shape) / 2.0
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), axis=-1)
    distance = np.linalg.norm(grid - centre, axis=-1)

    segmentation = ((distance > 16.0) & (distance < 20.0)).astype(np.float32)
    tomogram = np.zeros(shape, dtype=np.float32)
    tomogram[segmentation > 0] = -1.0
    tomogram[42:47, 28:33, 28:33] = -8.0

    segmentation_path = tmp_path / f'{tomogram_id}_seg.mrc'
    tomogram_path = tmp_path / f'{tomogram_id}_raw.mrc'
    with mrcfile.new(segmentation_path, overwrite=True) as mrc:
        mrc.set_data(segmentation)
    with mrcfile.new(tomogram_path, overwrite=True) as mrc:
        mrc.set_data(tomogram)

    return TomogramManifest(
        tomogram_id=tomogram_id,
        segmentation_path=segmentation_path,
        raw_tomogram_path=tomogram_path,
        voxel_size_angstrom=10.0,
    )

# _real_set: six fixed real particles inside the vesicle, 14 voxels from its centre
def _real_set(tomogram_id: str = 'tomo000') -> ParticleSet:
    positions = ((30, 30, 44), (30, 30, 16), (30, 44, 30), (30, 16, 30), (44, 30, 30), (16, 30, 30))
    return ParticleSet(
        particles=[
            Particle(
                particle_id=f'p{index:06d}',
                tomogram_id=tomogram_id,
                position=tuple(float(c) for c in position),
                source_picker='mine',
                half_set=HalfSet.A,
            )
            for index, position in enumerate(positions)
        ],
        consensus_rule='union',
        contributing_pickers=['mine'],
    )

# TestDecoyGenerate: class containing unit tests for decoy/generate.py
class TestDecoyGenerate:
    def test_shifted_decoys_stay_away_from_surface(self, tmp_path: Path) -> None:
        '''Shifted decoys stay inside the volume and off the surface'''
        manifest = _write_vesicle_pair(tmp_path)
        config = DecoyGeometry(voxel_size_angstrom=10.0)
        real_set = _real_set()
        decoy_set = generate_shifted_decoys(
            real_particle_set=real_set, manifests=[manifest], config=config,
            min_shift_angstrom=100.0, max_shift_angstrom=200.0,
            min_distance_from_surface_angstrom=50.0, min_pick_distance=60.0, seed=2,
        )
        for particle in decoy_set.particles:
            assert particle.source_picker == 'decoy-shifted'
            for coordinate in particle.position:
                assert 0 <= coordinate <= 59

    def test_shifted_decoys_are_oriented(self, tmp_path: Path) -> None:
        '''Shifted decoys have defined orientation.'''
        manifest = _write_vesicle_pair(tmp_path)
        config = DecoyGeometry(voxel_size_angstrom=10.0)
        real_set = _real_set()
        decoy_set = generate_shifted_decoys(
            real_particle_set=real_set, manifests=[manifest], config=config,
            min_shift_angstrom=100.0, max_shift_angstrom=200.0,
            min_distance_from_surface_angstrom=50.0,  min_pick_distance=60.0, seed=2,
        )
        for particle in decoy_set.particles:
            assert particle.orientation is not None

    def test_shifted_decoys_are_deterministic(self, tmp_path: Path) -> None:
        '''Same seed and parameters reproduce the same decoys'''
        manifest = _write_vesicle_pair(tmp_path)
        config = DecoyGeometry(voxel_size_angstrom=10.0)
        kwargs = dict(
            real_particle_set=_real_set(), manifests=[manifest], config=config,
            min_shift_angstrom=100.0, max_shift_angstrom=200.0,
            min_distance_from_surface_angstrom=50.0, min_pick_distance=60.0, seed=7,
        )
        first = generate_shifted_decoys(**kwargs)
        second = generate_shifted_decoys(**kwargs)
        assert [p.position for p in first.particles] == [p.position for p in second.particles]

    def test_synthetic_noise_writes_matched_pairs(self, tmp_path: Path) -> None:
        '''Synthetic-noise writes matched segmentation/tomogram pairs'''
        config = DecoyGeometry(voxel_size_angstrom=10.0)
        decoy_set, manifests = generate_synthetic_noise_decoys(
            tomogram_shape=(40, 40, 40), n_tomograms=2, n_decoys_per_tomogram=5,
            output_dir=tmp_path / 'noise', config=config, seed=3,
        )
        assert len(manifests) == 2
        for manifest in manifests:
            assert manifest.is_decoy
            assert manifest.segmentation_path.exists()
            assert manifest.raw_tomogram_path.exists()
            with mrcfile.open(manifest.segmentation_path) as mrc:
                assert mrc.data.shape == (40, 40, 40)
        assert len(decoy_set.particles) == 10

    def test_synthetic_decoys_are_oriented(self, tmp_path: Path) -> None:
        '''Synthetic decoys have defined orientation.'''
        config = DecoyGeometry(voxel_size_angstrom=10.0)
        decoy_set, manifests = generate_synthetic_noise_decoys(
            tomogram_shape=(40, 40, 40), n_tomograms=2, n_decoys_per_tomogram=5,
            output_dir=tmp_path / 'noise', config=config, seed=3,
        )
        assert all(p.orientation is not None for p in decoy_set.particles)

    def test_generate_synthetic_noise_decoys_requires_explicit_count(self, tmp_path):
        '''n_decoys_per_tomogram=None raises for synthetic noise decoys.'''
        config = DecoyGeometry(voxel_size_angstrom=10.0)
        real_set = _real_set()
        with pytest.raises(StampValidationError):
            generate_synthetic_noise_decoys(
                tomogram_shape=(40, 40, 40),
                n_tomograms=1,
                n_decoys_per_tomogram=None,
                output_dir=tmp_path,
                config=config,
                seed=1,
            )

    def test_decoy_geometry_rejects_bad_values(self) -> None:
        with pytest.raises(StampValidationError):
            DecoyGeometry(voxel_size_angstrom=0.0)
        with pytest.raises(StampValidationError):
            DecoyGeometry(voxel_size_angstrom=10.0, density_sign=0)

# TestDecoyValidate: class containing unit tests for decoy/validate.py
class TestDecoyValidate:
    def test_is_decoy_particle_set_detects_mixture(self) -> None:
        '''A mixed decoy/real set raises DecoyContaminationError'''
        def _particle(particle_id: str, picker: str) -> Particle:
            return Particle(
                particle_id=particle_id, tomogram_id='tomo000', position=(1.0, 1.0, 1.0),
                source_picker=picker, half_set=HalfSet.A,
            )

        mixed = ParticleSet(
            particles=[_particle('p1', 'stamp-native'), _particle('d1', 'decoy-shifted')],
            consensus_rule='union', contributing_pickers=['mixed'],
        )
        with pytest.raises(StampPipelineError, match='mixes decoy and real'):
            is_decoy_particle_set(mixed)

    def test_check_manifest_purity_rejects_mixture(self, tmp_path: Path) -> None:
        '''A mixed manifest list raises, a pure one returns its flag'''
        real = TomogramManifest(
            tomogram_id='a', segmentation_path=tmp_path / 'a.mrc',
            raw_tomogram_path=tmp_path / 'a_raw.mrc', voxel_size_angstrom=10.0,
        )
        fake = TomogramManifest(
            tomogram_id='b', segmentation_path=tmp_path / 'b.mrc',
            raw_tomogram_path=tmp_path / 'b_raw.mrc', voxel_size_angstrom=10.0, is_decoy=True,
        )
        with pytest.raises(StampPipelineError):
            check_manifest_purity([real, fake])
        assert check_manifest_purity([real]) is False
        assert check_manifest_purity([fake]) is True

    def test_assert_comparable_rejects_undersized_decoy_set(self) -> None:
        '''A decoy set under half the real size raises'''
        def _set(prefix: str, count: int, picker: str) -> ParticleSet:
            return ParticleSet(
                particles=[
                    Particle(
                        particle_id=f'{prefix}{i}', tomogram_id='tomo000',
                        position=(1.0, 1.0, 1.0), source_picker=picker, half_set=HalfSet.A,
                    )
                    for i in range(count)
                ],
                consensus_rule='union', contributing_pickers=[picker],
            )

        real = _set('p', 100, 'stamp-native')
        small_decoy = _set('d', 10, 'decoy-shifted')
        with pytest.raises(StampPipelineError, match='less than half the size'):
            assert_comparable(real, small_decoy)

        assert_comparable(real, _set('d', 80, 'decoy-shifted'))  # should not raise


# _real_set_for: a real particle set with the given prior and n particles clustered in the middle of a 60-voxel tomogram
def _real_set_for(prior, n: int = 12) -> ParticleSet:
    rng = np.random.default_rng(0)
    particles = [
        Particle(particle_id=f'p{i}', tomogram_id='tomo000', position=tuple(float(v) for v in rng.uniform(20, 40, 3)), source_picker='x', half_set=HalfSet.A if i % 2 else HalfSet.B)
        for i in range(n)
    ]
    return ParticleSet(particles=particles, consensus_rule='union', contributing_pickers=['x'], orientation_prior=prior)


# TestRandomVolumeDecoys: class containing tests for the prior-agnostic decoy method
class TestRandomVolumeDecoys:
    def test_avoids_real_picks_and_the_border(self, tmp_path: Path) -> None:
        from stamp.decoy.generate import generate_random_volume_decoys
        from scipy.spatial import cKDTree
        from stamp.schemas.particles import OrientationPrior
        manifest = _write_vesicle_pair(tmp_path)
        real = _real_set_for(OrientationPrior.NONE)
        decoys = generate_random_volume_decoys(real, [manifest], DecoyGeometry(voxel_size_angstrom=10.0), 60.0, 100.0, seed=1)
        positions = np.array([p.position for p in decoys.particles])
        assert len(positions) == len(real.particles)
        assert (cKDTree(np.array([p.position for p in real.particles])).query(positions)[0] >= 6.0).all()     # 60 A at 10 A/voxel
        assert (positions >= 10.0).all() and (positions <= 59 - 10.0).all()                                   # 100 A border
        assert decoys.orientation_prior is OrientationPrior.NONE and all(p.orientation is None for p in decoys.particles)
        assert all(p.source_picker == 'decoy-random-volume' for p in decoys.particles)

    def test_axis_prior_gets_random_axes_and_is_deterministic(self, tmp_path: Path) -> None:
        from stamp.decoy.generate import generate_random_volume_decoys
        from stamp.schemas.particles import OrientationPrior
        manifest = _write_vesicle_pair(tmp_path)
        real = _real_set_for(OrientationPrior.AXIS)
        first = generate_random_volume_decoys(real, [manifest], DecoyGeometry(voxel_size_angstrom=10.0), 60.0, 100.0, seed=2)
        second = generate_random_volume_decoys(real, [manifest], DecoyGeometry(voxel_size_angstrom=10.0), 60.0, 100.0, seed=2)
        assert all(p.orientation is not None for p in first.particles) and len({p.orientation for p in first.particles}) > 1
        assert first == second

    def test_refuses_the_membrane_prior(self, tmp_path: Path) -> None:
        from stamp.decoy.generate import generate_random_volume_decoys
        from stamp.schemas.particles import OrientationPrior
        with pytest.raises(StampValidationError, match='shifted'):
            generate_random_volume_decoys(_real_set_for(OrientationPrior.NORMAL), [_write_vesicle_pair(tmp_path)], DecoyGeometry(voxel_size_angstrom=10.0), 60.0, 100.0, seed=0)

    def test_comparable_requires_the_same_prior(self, tmp_path: Path) -> None:
        from stamp.decoy.generate import generate_random_volume_decoys
        from stamp.schemas.particles import OrientationPrior
        real = _real_set_for(OrientationPrior.NONE)
        decoys = generate_random_volume_decoys(real, [_write_vesicle_pair(tmp_path)], DecoyGeometry(voxel_size_angstrom=10.0), 60.0, 100.0, seed=1)
        assert_comparable(real, decoys)
        with pytest.raises(StampPipelineError, match='orientation prior'):
            assert_comparable(real.model_copy(update={'orientation_prior': OrientationPrior.AXIS}), decoys)
