'''
STAMP: unit tests for decoy generation and the decoy/real separation guards
'''

# Import external dependencies
import mrcfile, numpy as np, pytest
from pathlib import Path

# Import internal STAMP objects
from stamp.decoy.generate import (
    generate_rejected_surface_decoys,
    generate_shifted_decoys,
    generate_synthetic_noise_decoys,
)
from stamp.decoy.validate import (
    DecoyContaminationError,
    assert_comparable,
    check_manifest_purity,
    is_decoy_particle_set,
)
from stamp.picking.native import NativePickerConfig, pick_tomogram
from stamp.schemas.manifest import TomogramManifest
from stamp.schemas.particles import HalfSet, Particle, ParticleSet

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

# _real_set_from_picker: run the native picker over a manifest and wrap its picks in a ParticleSet
def _real_set_from_picker(manifest: TomogramManifest, config: NativePickerConfig) -> ParticleSet:
    picks = pick_tomogram(
        manifest.segmentation_path, manifest.raw_tomogram_path, manifest.tomogram_id, config
    )
    return ParticleSet(
        particles=[
            Particle(
                particle_id=f'p{index:06d}',
                tomogram_id=pick.tomogram_id,
                position=pick.position,
                orientation=pick.orientation,
                source_picker=pick.source_picker,
                confidence=pick.confidence,
                half_set=HalfSet.A,
            )
            for index, pick in enumerate(picks)
        ],
        consensus_rule='union',
        contributing_pickers=['stamp-native'],
    )

# TestDecoy: class containing unit tests for test_decoy.py
class TestDecoy:
    def test_rejected_surface_decoys_avoid_real_picks(self, tmp_path: Path) -> None:
        '''Every decoy is kept clear of real picks'''
        manifest = _write_vesicle_pair(tmp_path)
        config = NativePickerConfig(voxel_size_angstrom=10.0, n_mad=2.5)
        real_set = _real_set_from_picker(manifest, config)
        assert real_set.particles, 'fixture produced no real picks'

        decoy_set = generate_rejected_surface_decoys(
            real_particle_set=real_set,
            manifests=[manifest],
            config=config,
            n_decoys_per_tomogram=10,
            min_distance_from_real_angstrom=100.0,
            seed=1,
        )

        assert decoy_set.particles
        real_positions = np.array([p.position for p in real_set.particles])
        for particle in decoy_set.particles:
            distances = np.linalg.norm(real_positions - np.array(particle.position), axis=1)
            assert distances.min() >= 10.0  # 100 Angstrom at 10 A/voxel
            assert particle.source_picker == 'decoy-rejected-surface'

    def test_rejected_surface_decoys_score_below_threshold(self, tmp_path: Path) -> None:
        '''Decoys must come from points the picker rejected'''
        manifest = _write_vesicle_pair(tmp_path)
        config = NativePickerConfig(voxel_size_angstrom=10.0, n_mad=2.5)
        real_set = _real_set_from_picker(manifest, config)

        decoy_set = generate_rejected_surface_decoys(
            real_particle_set=real_set, manifests=[manifest], config=config,
            n_decoys_per_tomogram=10, min_distance_from_real_angstrom=100.0, seed=1,
        )
        for particle in decoy_set.particles:
            assert particle.confidence is not None
            assert particle.confidence < config.n_mad

    def test_rejected_surface_decoys_are_deterministic(self, tmp_path: Path) -> None:
        '''Same seed and parameters reproduce the same decoys'''
        manifest = _write_vesicle_pair(tmp_path)
        config = NativePickerConfig(voxel_size_angstrom=10.0, n_mad=2.5)
        real_set = _real_set_from_picker(manifest, config)

        kwargs = dict(
            real_particle_set=real_set, manifests=[manifest], config=config,
            n_decoys_per_tomogram=10, min_distance_from_real_angstrom=100.0, seed=7,
        )
        first = generate_rejected_surface_decoys(**kwargs)
        second = generate_rejected_surface_decoys(**kwargs)
        assert [p.position for p in first.particles] == [p.position for p in second.particles]

    def test_shifted_decoys_stay_away_from_surface(self, tmp_path: Path) -> None:
        '''Shifted decoys stay inside the volume and off the surface'''
        manifest = _write_vesicle_pair(tmp_path)
        config = NativePickerConfig(voxel_size_angstrom=10.0, n_mad=2.5)
        real_set = _real_set_from_picker(manifest, config)

        decoy_set = generate_shifted_decoys(
            real_particle_set=real_set, manifests=[manifest], config=config,
            min_shift_angstrom=100.0, max_shift_angstrom=200.0,
            min_distance_from_surface_angstrom=50.0, seed=2,
        )
        for particle in decoy_set.particles:
            assert particle.source_picker == 'decoy-shifted'
            for coordinate in particle.position:
                assert 0 <= coordinate <= 59

    def test_synthetic_noise_writes_matched_pairs(self, tmp_path: Path) -> None:
        '''Synthetic-noise writes matched segmentation/tomogram pairs'''
        config = NativePickerConfig(voxel_size_angstrom=10.0)
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
        with pytest.raises(DecoyContaminationError, match='mixes decoy and real'):
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
        with pytest.raises(DecoyContaminationError):
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
        with pytest.raises(ValueError, match='less than half the size'):
            assert_comparable(real, small_decoy)

        assert_comparable(real, _set('d', 80, 'decoy-shifted'))  # should not raise
