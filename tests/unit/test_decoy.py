'''
STAMP: unit tests for decoy generation and the decoy/real separation guards
'''

# Import external dependencies
import mrcfile, numpy as np, pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.decoy.generate import DecoyGeometry, generate_random_volume_decoys, generate_shifted_decoys, generate_synthetic_noise_decoys
from stamp.decoy.validate import (
    assert_comparable,
    check_manifest_purity,
    is_decoy_particle_set,
)
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.particles import HalfSet, OrientationPrior, Particle, ParticleSet
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
                prior=OrientationPrior.NORMAL,
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

# _mixed_set: real particles at the given (position, prior) pairs in one tomogram
def _mixed_set(entries: list[tuple[tuple[float, float, float], OrientationPrior]]) -> ParticleSet:
    particles = [
        Particle(
            particle_id=f'p{i}', tomogram_id='tomo000', position=position, prior=prior,
            orientation=(1.0, 0.0, 0.0, 0.0) if prior is not OrientationPrior.NONE else None,
            source_picker='mine', half_set=HalfSet.A if i % 2 == 0 else HalfSet.B,
        )
        for i, (position, prior) in enumerate(entries)
    ]
    return ParticleSet(particles=particles, consensus_rule='union', contributing_pickers=['mine'])

# _decoy_like: a decoy ParticleSet built from the real particles whose prior is in keep, truncated to n
def _decoy_like(real: ParticleSet, keep: set[OrientationPrior], n: int | None = None) -> ParticleSet:
    particles = [
        Particle(
            particle_id=f'decoy-{i:06d}', tomogram_id=p.tomogram_id, position=p.position, prior=p.prior,
            orientation=p.orientation, source_picker='decoy-test', half_set=HalfSet.A if i % 2 == 0 else HalfSet.B,
        )
        for i, p in enumerate(p for p in real.particles if p.prior in keep)
    ][:n]
    return ParticleSet(particles=particles, consensus_rule='union', contributing_pickers=['decoy-test'])

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
                prior=OrientationPrior.NORMAL,
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
                        prior=OrientationPrior.NORMAL,
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
        Particle(prior=prior, particle_id=f'p{i}', tomogram_id='tomo000', position=tuple(float(v) for v in rng.uniform(20, 40, 3)), source_picker='x', half_set=HalfSet.A if i % 2 else HalfSet.B)
        for i in range(n)
    ]
    return ParticleSet(particles=particles, consensus_rule='union', contributing_pickers=['x'])


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
        assert list(decoys.by_prior()) == [OrientationPrior.NONE] and all(p.orientation is None for p in decoys.particles)
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
        with pytest.raises(StampPipelineError, match='real set lacks'):
            assert_comparable(real.model_copy(update={'particles': [p.model_copy(update={'prior': OrientationPrior.AXIS}) for p in real.particles]}), decoys)

# TestMixedPriorDecoys: class containing tests for decoys built from particle sets with several priors
class TestMixedPriorDecoys:
    def test_random_volume_covers_axis_and_none_only(self, tmp_path: Path) -> None:
        '''Random-volume gives one decoy per axis or none particle, with that prior, and none for normal'''
        manifest = _write_vesicle_pair(tmp_path)
        real = _mixed_set([((20.0, 20.0, 20.0), OrientationPrior.AXIS), ((30.0, 30.0, 30.0), OrientationPrior.NONE), ((40.0, 40.0, 40.0), OrientationPrior.NORMAL)])
        decoys = generate_random_volume_decoys(real, [manifest], DecoyGeometry(voxel_size_angstrom=10.0), 20.0, 5.0, seed=1)
        assert sorted(p.prior.value for p in decoys.particles) == ['axis', 'none']
        assert all(p.orientation is not None for p in decoys.particles if p.prior is OrientationPrior.AXIS)
        assert all(p.orientation is None for p in decoys.particles if p.prior is OrientationPrior.NONE)

    def test_shifted_displaces_normal_particles_only(self, tmp_path: Path) -> None:
        '''Shifted decoys come from normal particles only, while axis particles still exclude decoy positions'''
        manifest = _write_vesicle_pair(tmp_path)
        real = _mixed_set([((30.0, 30.0, 44.0), OrientationPrior.NORMAL), ((30.0, 30.0, 16.0), OrientationPrior.AXIS)])
        decoys = generate_shifted_decoys(
            real_particle_set=real, manifests=[manifest], config=DecoyGeometry(voxel_size_angstrom=10.0),
            min_shift_angstrom=100.0, max_shift_angstrom=180.0,
            min_distance_from_surface_angstrom=50.0, min_pick_distance=60.0, seed=2,
        )
        assert {p.prior for p in decoys.particles} == {OrientationPrior.NORMAL}
        assert len(decoys.particles) == 1

    def test_random_volume_rejects_a_normal_only_set(self, tmp_path: Path) -> None:
        '''Random-volume with nothing it supports raises and points at the shifted method'''
        manifest = _write_vesicle_pair(tmp_path)
        with pytest.raises(StampValidationError, match='shifted'):
            generate_random_volume_decoys(_mixed_set([((20.0, 20.0, 20.0), OrientationPrior.NORMAL)]), [manifest], DecoyGeometry(voxel_size_angstrom=10.0), 20.0, 5.0, seed=1)

# TestDecoyComparablePerPrior: class containing tests for per-prior comparison of real and decoy sets
class TestDecoyComparablePerPrior:
    def test_decoy_may_cover_fewer_priors(self) -> None:
        '''A decoy covering one of two real priors is comparable'''
        real = _mixed_set([((1.0, 1.0, 1.0), OrientationPrior.NORMAL), ((2.0, 2.0, 2.0), OrientationPrior.NONE)])
        assert_comparable(real, _decoy_like(real, keep={OrientationPrior.NORMAL}))

    def test_each_covered_prior_is_size_checked(self) -> None:
        '''A normal decoy covering one of ten real normal particles fails even when the totals look fine'''
        real = _mixed_set([((float(i), 1.0, 1.0), OrientationPrior.NORMAL) for i in range(10)] + [((1.0, 2.0, 2.0), OrientationPrior.NONE)])
        with pytest.raises(StampPipelineError, match='real normal set'):
            assert_comparable(real, _decoy_like(real, keep={OrientationPrior.NORMAL}, n=1))

# TestAutoMethod: class containing tests for --method auto resolution and merging
class TestAutoMethod:
    def test_resolve_methods_follows_the_priors_present(self) -> None:
        '''auto maps each prior to its method, and an explicit method passes through'''
        from stamp.decoy.generate import METHOD_AUTO, METHOD_RANDOM_VOLUME, METHOD_SHIFTED, resolve_methods
        assert resolve_methods(METHOD_AUTO, {OrientationPrior.NORMAL, OrientationPrior.NONE}) == [METHOD_RANDOM_VOLUME, METHOD_SHIFTED]
        assert resolve_methods(METHOD_AUTO, {OrientationPrior.AXIS}) == [METHOD_RANDOM_VOLUME]
        assert resolve_methods(METHOD_SHIFTED, {OrientationPrior.NONE}) == [METHOD_SHIFTED]

    def test_merge_decoy_sets_keeps_ids_unique(self, tmp_path: Path) -> None:
        '''Merged decoys get fresh unique ids and keep the method that made each one'''
        from stamp.decoy.generate import generate_random_volume_decoys, merge_decoy_sets
        manifest = _write_vesicle_pair(tmp_path)
        config = DecoyGeometry(voxel_size_angstrom=10.0)
        real = _mixed_set([((30.0, 30.0, 44.0), OrientationPrior.NORMAL), ((30.0, 30.0, 16.0), OrientationPrior.NONE)])
        shifted = generate_shifted_decoys(
            real_particle_set=real, manifests=[manifest], config=config,
            min_shift_angstrom=100.0, max_shift_angstrom=180.0,
            min_distance_from_surface_angstrom=50.0, min_pick_distance=60.0, seed=2,
        )
        volume = generate_random_volume_decoys(real, [manifest], config, 60.0, 50.0, seed=1)
        merged = merge_decoy_sets([shifted, volume], seed=1)
        ids = [p.particle_id for p in merged.particles]
        assert len(set(ids)) == len(ids) == len(shifted.particles) + len(volume.particles)
        assert {p.source_picker for p in merged.particles} == {'decoy-shifted', 'decoy-random-volume'}
        assert {p.prior for p in merged.particles} == {OrientationPrior.NORMAL, OrientationPrior.NONE}


# TestPerTomogramSeeding: class checking that one tomogram's decoy placement does not depend on the other tomograms in the run
class TestPerTomogramSeeding:
    def _positions(self, decoys: ParticleSet | None, tomogram_id: str) -> list[tuple[float, float, float]]:
        assert decoys is not None
        return [p.position for p in decoys.particles if p.tomogram_id == tomogram_id]

    def test_shifted_placement_independent_of_other_tomograms(self, tmp_path: Path) -> None:
        manifest_a = _write_vesicle_pair(tmp_path, 'tomoA')
        manifest_b = _write_vesicle_pair(tmp_path, 'tomoB')
        config = DecoyGeometry(voxel_size_angstrom=10.0)
        kwargs = dict(config=config, min_shift_angstrom=100.0, max_shift_angstrom=200.0, min_distance_from_surface_angstrom=50.0, min_pick_distance=60.0, seed=5)
        alone = generate_shifted_decoys(real_particle_set=_real_set('tomoB'), manifests=[manifest_b], **kwargs)
        both = generate_shifted_decoys(real_particle_set=_real_set('tomoA').model_copy(update={'particles': _real_set('tomoA').particles + _real_set('tomoB').particles}), manifests=[manifest_a, manifest_b], **kwargs)
        assert self._positions(alone, 'tomoB') and self._positions(alone, 'tomoB') == self._positions(both, 'tomoB')

    def test_random_volume_placement_independent_of_other_tomograms(self, tmp_path: Path) -> None:
        manifest_a = _write_vesicle_pair(tmp_path, 'tomoA')
        manifest_b = _write_vesicle_pair(tmp_path, 'tomoB')
        config = DecoyGeometry(voxel_size_angstrom=10.0)
        real_b = _real_set_for(OrientationPrior.AXIS).model_copy(update={'particles': [p.model_copy(update={'tomogram_id': 'tomoB'}) for p in _real_set_for(OrientationPrior.AXIS).particles]})
        real_a = _real_set_for(OrientationPrior.AXIS).model_copy(update={'particles': [p.model_copy(update={'tomogram_id': 'tomoA'}) for p in _real_set_for(OrientationPrior.AXIS).particles]})
        alone = generate_random_volume_decoys(real_b, [manifest_b], config, 60.0, 100.0, seed=5)
        both = generate_random_volume_decoys(real_a.model_copy(update={'particles': real_a.particles + real_b.particles}), [manifest_a, manifest_b], config, 60.0, 100.0, seed=5)
        assert self._positions(alone, 'tomoB') and self._positions(alone, 'tomoB') == self._positions(both, 'tomoB')
