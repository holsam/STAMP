'''
STAMP: unit tests for subvolume extraction, feature classification, and clustering
'''

# Import external dependencies
import mrcfile, numpy as np, pytest
from pathlib import Path
from scipy.ndimage import gaussian_filter, rotate

# Import internal STAMP objects
from stamp.classify.align import roll_about_normal
from stamp.classify.cluster import ClusteringConfig, label_to_cluster_id, reduce_and_cluster, reduce_features
from stamp.classify.extract import box_fits_inside, canonical_grid, extract_particle_set, extract_subvolume, quaternion_to_matrix
from stamp.classify.features import azimuthal_magnitudes, build_feature_matrix, cylindrical_bins, rotational_average
from stamp.classify.frames import azimuth_degrees, azimuth_matrix, flip_matrix, rotate_density, rotation_to_quaternion
from stamp.classify.match import align_to_reference, rank_rotations, wedge_filter
from stamp.classify.merge import half_split_ceilings, merge_groups, pair_thresholds, pairwise_similarity
from stamp.classify.null import null_bar, null_threshold
from stamp.classify.polish import PolishConfig, run_polish
from stamp.classify.prepare import fourier_bin, normalise, prepare_particle, soft_mask, working_box
from stamp.classify.references import hard_assign, rebuild_references, score_against_refs
from stamp.classify.search import rotated_bank, search_space
from stamp.classify.store import build_store
from stamp.classify.wedge import apply_wedge, box_wedge, observed_directions
from stamp.picking.geometry import quaternion_from_reference_to
from stamp.schemas.particles import HalfSet, OrientationPrior, Particle
from stamp.schemas.subvolumes import Subvolumes
from synthetic import asymmetric_object, make_particle, rod_object, synthetic_set

# _BOX: working box edge for alignment tests
_BOX = 25

# _BOX: working box edge for reference scoring tests
_REF_BOX = 21


# _aligned_cc: normalised correlation of two volumes
def _aligned_cc(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))

# _angle_error: absolute difference of two angles in degrees, wrapped to [0, 180]
def _angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)

# _asymmetric_blob: a subvolume with a clear azimuthal feature (an off-axis lobe)
def _asymmetric_blob(box: int) -> np.ndarray:
    volume = np.zeros((box, box, box))
    volume[box // 2 + 4, box // 2, box // 2] = 1.0
    volume[box // 2, box // 2, box // 2] = 1.0
    return gaussian_filter(volume, 1.5)

# _blob: volume with one gaussian blob at the given integer index
def _blob(shape: tuple[int, int, int], index: tuple[int, int, int]) -> np.ndarray:
    volume = np.zeros(shape, dtype=np.float32)
    volume[index] = 1.0
    return gaussian_filter(volume, 1.0)

# _brute_wedge: explicit triple loop, tomogram-frame vectors from the rotation matrix, tilt about the axis at tilt_axis_degrees
def _brute_wedge(box: int, orientation, tilt_max: float, tilt_axis: float) -> np.ndarray:
    rotation = np.eye(3) if orientation is None else quaternion_to_matrix(orientation)
    axis = np.radians(tilt_axis)
    direction = np.array([np.cos(axis), np.sin(axis), 0.0])     # tilt axis itself
    freq = np.fft.fftfreq(box)
    out = np.zeros((box,) * 3, dtype=np.float32)
    for i in range(box):
        for j in range(box):
            for k in range(box):
                tomo = rotation @ np.array([freq[i], freq[j], freq[k]])
                across = np.sqrt(max(tomo[0] ** 2 + tomo[1] ** 2 - (tomo @ direction) ** 2, 0.0))   # in-plane component perpendicular to the tilt axis
                out[i, j, k] = abs(tomo[2]) <= across * np.tan(np.radians(tilt_max)) + 1e-12
    return out

# _c_n_particle: a ring of n_fold blobs above the canonical mid-plane (idealised C-n oligomer on a membrane)
def _c_n_particle(box: int = 25, n_fold: int = 4, radius: float = 6.0) -> np.ndarray:
    volume = np.zeros((box, box, box))
    centre = (box - 1) / 2.0
    z_plane = int(centre + 4)

    for index in range(n_fold):
        angle = 2.0 * np.pi * index / n_fold
        x = int(round(centre + radius * np.cos(angle)))
        y = int(round(centre + radius * np.sin(angle)))
        volume[x - 1 : x + 2, y - 1 : y + 2, z_plane - 1 : z_plane + 2] = 1.0
    return volume

# _concatenate: one store from the volumes of two synthetic sets
def _concatenate(directory: Path, first, second):
    from stamp.classify.store import ParticleStore
    from stamp.schemas.subvolumes import Subvolumes
    directory.mkdir(parents=True)
    volumes = [np.stack([s.store.volumes[i] for i in range(len(s.store))]) for s in (first, second)]
    np.save(directory / 'all.npy', np.concatenate(volumes))
    total = sum(len(v) for v in volumes)
    return ParticleStore(Subvolumes(directory, [('all', i) for i in range(total)], _BOX), first.store.orientations + second.store.orientations, first.store.tilt_max_degrees, first.store.tilt_axis_degrees)

# _config: small polish settings for fast tests
def _config(prior: OrientationPrior, **overrides) -> PolishConfig:
    return PolishConfig(prior=prior, angular_step_degrees=15.0, n_rotations=250, shift_range_voxels=2, min_seed_size=10, **overrides)

# _corrupted_seeds: true labels with a fraction replaced by a random class, plus an embedding in which seeds sit near their class
def _corrupted_seeds(truth: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    seeds = truth.copy()
    wrong = rng.random(len(truth)) < fraction
    seeds[wrong] = rng.integers(0, truth.max() + 1, wrong.sum())
    embedding = np.eye(truth.max() + 1)[seeds] + rng.normal(0, 0.1, (len(truth), truth.max() + 1))
    return seeds, embedding

# _peak: index of the maximum of a volume
def _peak(volume: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.unravel_index(volume.argmax(), volume.shape))

# _purity: fraction of accepted particles whose cluster's majority class is their own class
def _purity(labels: np.ndarray, truth: np.ndarray) -> float:
    accepted = labels >= 0
    hits = sum(np.bincount(truth[labels == k]).max() for k in np.unique(labels[accepted]))
    return float(hits / accepted.sum())

# _random_orientation: unit quaternion mapping +z onto a random direction within 60 degrees of the beam-orthogonal plane
def _random_orientation(rng: np.random.Generator) -> tuple[float, float, float, float]:
    vector = rng.normal(size=3)
    vector[2] *= 0.5
    return quaternion_from_reference_to(vector / np.linalg.norm(vector))

# _rotate_in_plane: rotate a canonical (x, y, z) volume in-plane about axes (0, 1)
def _rotate_in_plane(volume: np.ndarray, degrees: float) -> np.ndarray:
    return rotate(volume, angle=degrees, axes=(0, 1), reshape=False, order=1)

# _subvols_from_array: write a dense subvolume stack to a single-tomogram cache and wrap it as a Subvolumes subvols
def _subvols_from_array(tmp_path: Path, array: np.ndarray) -> Subvolumes:
    tmp_path.mkdir(parents=True, exist_ok=True)
    box_voxels = array.shape[-1]
    np.save(tmp_path / 'tomo000.npy', np.asarray(array, dtype=np.float32))
    index = [('tomo000', i) for i in range(array.shape[0])]
    return Subvolumes(tmp_path, index, box_voxels)

# _three_blobs: three well-separated Gaussian blobs in 2D
def _three_blobs(n_per_group: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
    return np.vstack(
        [rng.normal(centre, 1.0, size=(n_per_group, 2)) for centre in centres]
    )

# _nearest_neighbour_accuracy: fraction of rows whose nearest other row has the same label
def _nearest_neighbour_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    distances = np.linalg.norm(features[:, None] - features[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float((labels[distances.argmin(axis=1)] == labels).mean())

# TestAverage: class containing tests for subvolume averages
class TestAverage:
    # test_weighted_average_beats_plain_mean: on wedged particles of known pose the coverage-corrected average is closer to the truth than the plain mean of the aligned particles, for every prior
    @pytest.mark.parametrize('prior', ['normal', 'axis', 'none'])
    def test_weighted_average_beats_plain_mean(self, prior: str) -> None:
        from stamp.classify.average import weighted_average
        from stamp.classify.frames import azimuth_matrix, flip_matrix, rotate_density
        from stamp.classify.wedge import box_wedge
        from scipy.spatial.transform import Rotation
        from synthetic import asymmetric_object, make_particle, rod_object
        box, n = 25, 60
        rng = np.random.default_rng(8)
        obj = rod_object(box, 1, repeat=box) if prior == 'axis' else asymmetric_object(box, 1)
        if prior == 'none':
            rotations = Rotation.random(n, random_state=1).as_matrix()
            wedges = np.stack([box_wedge(box, None, 35.0, 90.0)] * n)
        else:
            flips = [flip_matrix() if prior == 'axis' and rng.random() < 0.5 else np.eye(3) for _ in range(n)]
            rotations = np.stack([azimuth_matrix(a) @ f for a, f in zip(rng.uniform(0, 360, n), flips)])
            wedges = np.stack([box_wedge(box, quaternion_from_reference_to(rng.normal(size=3)), 35.0, 90.0) for _ in range(n)])
        shifts = rng.integers(-2, 3, (n, 3)) * (np.array([0, 0, 1]) if prior == 'axis' else 1)
        particles = np.stack([make_particle(obj, r, tuple(s), w, 0.02, rng) for r, s, w in zip(rotations, shifts, wedges)])
        weighted = weighted_average(particles, wedges, rotations, shifts)
        plain = np.mean([rotate_density(np.roll(p, -s, axis=(0, 1, 2)), r.T) for p, r, s in zip(particles, rotations, shifts)], axis=0)
        assert _aligned_cc(weighted, obj) > _aligned_cc(plain, obj)

# TestCluster: class containing unit tests for src/stamp/classify/cluster.py
class TestCluster:
    def test_hdbscan_recovers_three_groups(self) -> None:
        '''HDBSCAN finds three groups in three-blob data.'''
        result = reduce_and_cluster(
            _three_blobs(),
            ClusteringConfig(method='hdbscan', n_components=2, min_cluster_size=10),
        )
        real_labels = {label for label in result.labels if label != -1}
        assert len(real_labels) == 3

    def test_kmeans_respects_requested_cluster_count(self) -> None:
        '''KMeans returns exactly n_clusters groups.'''
        result = reduce_and_cluster(
            _three_blobs(), ClusteringConfig(method='kmeans', n_components=2, n_clusters=3)
        )
        assert len(set(result.labels)) == 3

    def test_pca_components_capped_by_sample_count(self) -> None:
        '''N_components is capped at what the data supports.'''
        features = np.random.default_rng(0).normal(size=(5, 100))
        embedding, explained = reduce_features(features, ClusteringConfig(n_components=50))
        assert embedding.shape[1] <= 4
        assert explained.shape == (embedding.shape[1],)

    def test_too_few_particles_raises(self) -> None:
        '''A single particle raises a clear error.'''
        with pytest.raises(ValueError, match='Too few particles'):
            reduce_features(np.zeros((1, 10)), ClusteringConfig())

    def test_hdbscan_below_min_cluster_size_raises_clearly(self) -> None:
        '''Too few particles for min_cluster_size raises.'''
        with pytest.raises(ValueError, match='min_cluster_size'):
            reduce_and_cluster(
                np.random.default_rng(0).normal(size=(5, 10)),
                ClusteringConfig(method='hdbscan', min_cluster_size=20),
            )

    def test_noise_label_becomes_named_id(self) -> None:
        '''Label -1 becomes 'noise', other labels become 'cNN'.'''
        assert label_to_cluster_id(-1) == 'noise'
        assert label_to_cluster_id(3) == 'c03'

    # test_leiden_finds_blobs_and_applies_min_size: three well separated blobs give three clusters, and communities under min_cluster_size are noise
    def test_leiden_finds_blobs_and_applies_min_size(self) -> None:
        rng = np.random.default_rng(0)
        centres = np.array([[0.0] * 5, [8.0] * 5, [-8.0, 8.0, -8.0, 8.0, -8.0]])
        features = np.vstack([rng.normal(c, 1.0, (40, 5)) for c in centres])
        result = reduce_and_cluster(features, ClusteringConfig(method='leiden', n_components=5, min_cluster_size=10))
        assert result.labels.max() == 2 and all(len(set(result.labels[i * 40:(i + 1) * 40])) == 1 for i in range(3))
        assert set(result.centroids) == {0, 1, 2}
        assert (reduce_and_cluster(features, ClusteringConfig(method='leiden', n_components=5, min_cluster_size=50)).labels == -1).all()

    # test_leiden_is_reproducible: same seed, same partition
    def test_leiden_is_reproducible(self) -> None:
        features = np.random.default_rng(1).normal(size=(120, 6))
        config = ClusteringConfig(method='leiden', n_components=4, min_cluster_size=5, random_state=3)
        assert np.array_equal(reduce_and_cluster(features, config).labels, reduce_and_cluster(features, config).labels)

# TestExtract: class containing unit tests for src/stamp/classify/extract.py
class TestExtract:
    def test_identity_quaternion_gives_identity_matrix(self) -> None:
        '''The unit quaternion maps to the identity matrix.'''
        np.testing.assert_allclose(quaternion_to_matrix((1.0, 0.0, 0.0, 0.0)), np.eye(3), atol=1e-9)

    @pytest.mark.parametrize('normal', [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [-1, 0.5, 0.2]])
    def test_quaternion_matrix_round_trip(self, normal) -> None:
        '''The matrix must send +z back to the normal the quaternion encoded.'''
        target = np.array(normal, dtype=float)
        target /= np.linalg.norm(target)
        quaternion = quaternion_from_reference_to(target)
        rotated = quaternion_to_matrix(quaternion) @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(rotated, target, atol=1e-6)

    def test_canonical_grid_is_centred(self) -> None:
        '''The sampling grid is centred on the origin.'''
        grid = canonical_grid(5)
        assert grid.shape == (3, 125)
        np.testing.assert_allclose(grid.mean(axis=1), 0.0, atol=1e-9)

    def test_extraction_centres_on_a_point_source(self) -> None:
        '''A point source lands at the subvolume centre.'''
        volume = np.zeros((40, 40, 40), dtype=np.float32)
        volume[20, 20, 20] = 1.0  # (z, y, x)
        subvolume = extract_subvolume(volume, (20.0, 20.0, 20.0), None, 9)
        centre = subvolume[4, 4, 4]
        assert centre == pytest.approx(1.0)
        assert subvolume.sum() == pytest.approx(1.0)

    def test_rotation_aligns_feature_to_canonical_z(self) -> None:
        '''A blob offset along +x in the volume should appear along +z in the frame when the normal points along +x.'''
        volume = np.zeros((60, 60, 60), dtype=np.float32)
        volume[30, 30, 36] = 1.0  # (z, y, x): +6 in x
        normal = np.array([1.0, 0.0, 0.0])
        quaternion = quaternion_from_reference_to(normal)
        subvolume = extract_subvolume(volume, (30.0, 30.0, 30.0), quaternion, 21)
        peak = np.unravel_index(np.argmax(subvolume), subvolume.shape)
        # Frame is indexed (x, y, z); the blob should sit at +6 in z.
        assert peak[2] > 10, f'expected peak beyond canonical centre in z, got {peak}'
        assert abs(peak[0] - 10) <= 1
        assert abs(peak[1] - 10) <= 1

    def test_box_fits_inside_uses_circumscribing_radius(self) -> None:
        '''Near-face particles are rejected for any orientation.'''
        assert box_fits_inside((30.0, 30.0, 30.0), (60, 60, 60), 11) is True
        assert box_fits_inside((2.0, 30.0, 30.0), (60, 60, 60), 11) is False

    def test_membrane_voxels_replaced_with_background(self, tmp_path) -> None:
        shape = (40, 40, 40)
        rng = np.random.default_rng(0)
        tomogram = rng.normal(0.0, 1.0, shape).astype(np.float32)
        tomogram[:, :, 20] += 50.0  # bright membrane plane
        segmentation = np.zeros(shape, dtype=np.float32)
        segmentation[:, :, 20] = 1.0

        with mrcfile.new(tmp_path / 't.mrc', overwrite=True) as mrc:
            mrc.set_data(tomogram)
            mrc.voxel_size = 1.0
        with mrcfile.new(tmp_path / 's.mrc', overwrite=True) as mrc:
            mrc.set_data(segmentation)
            mrc.voxel_size = 1.0

        particle = Particle(particle_id='p0', tomogram_id='t', position=(20.0, 20.0, 20.0), orientation=(1.0, 0.0, 0.0, 0.0), source_picker='test', confidence=1.0, half_set=HalfSet.A)
        subvolumes, kept, _ = extract_particle_set([particle], {'t': str(tmp_path / 't.mrc')}, 11, segmentation_paths={'t': str(tmp_path / 's.mrc')}, cache_dir=tmp_path)
        assert len(kept) == 1
        # the box centre sampled the membrane plane; it must now sit near background
        assert abs(float(subvolumes[0][5, 5, 5])) < 5.0

    def test_no_segmentation_leaves_tomogram_untouched(self, tmp_path) -> None:
        shape = (40, 40, 40)
        tomogram = np.zeros(shape, dtype=np.float32)
        tomogram[:, :, 20] = 50.0
        with mrcfile.new(tmp_path / 't.mrc', overwrite=True) as mrc:
            mrc.set_data(tomogram)
            mrc.voxel_size = 1.0
        particle = Particle(particle_id='p0', tomogram_id='t', position=(20.0, 20.0, 20.0), orientation=(1.0, 0.0, 0.0, 0.0), source_picker='test', confidence=1.0, half_set=HalfSet.A)
        subvolumes, kept, _ = extract_particle_set([particle], {'t': str(tmp_path / 't.mrc')}, 11, cache_dir=tmp_path)
        assert float(subvolumes[0][5, 5, 5]) == 50.0

    def test_shape_mismatch_falls_back_to_raw_tomogram(self, tmp_path, caplog) -> None:
        tomogram = np.zeros((40, 40, 40), dtype=np.float32)
        tomogram[:, :, 20] = 50.0
        segmentation = np.zeros((30, 30, 30), dtype=np.float32)  # wrong shape
        with mrcfile.new(tmp_path / 't.mrc', overwrite=True) as mrc:
            mrc.set_data(tomogram)
            mrc.voxel_size = 1.0
        with mrcfile.new(tmp_path / 's.mrc', overwrite=True) as mrc:
            mrc.set_data(segmentation)
            mrc.voxel_size = 1.0
        particle = Particle(particle_id='p0', tomogram_id='t', position=(20.0, 20.0, 20.0), orientation=(1.0, 0.0, 0.0, 0.0), source_picker='test', confidence=1.0, half_set=HalfSet.A)
        subvolumes, kept, skipped = extract_particle_set([particle], {'t': str(tmp_path / 't.mrc')}, 11, segmentation_paths={'t': str(tmp_path / 's.mrc')}, cache_dir=tmp_path)
        assert len(kept) == 1 and not skipped
        assert float(subvolumes[0][5, 5, 5]) == 50.0  # raw density, membrane untouched
        assert not (tmp_path / '.subtracted').exists()
        assert 'without membrane subtraction' in caplog.text
        assert '1 (raw density)' in caplog.text

    def test_aborted_extraction_resumes_without_recomputing(self, tmp_path, monkeypatch) -> None:
        from stamp.classify import extract as extract_module
        particles, paths = [], {}
        for name in ('a', 'b'):
            with mrcfile.new(tmp_path / f'{name}.mrc', overwrite=True) as mrc:
                mrc.set_data(np.random.default_rng(0).normal(size=(40, 40, 40)).astype(np.float32))
            paths[name] = str(tmp_path / f'{name}.mrc')
            particles.append(Particle(particle_id=f'p_{name}', tomogram_id=name, position=(20.0, 20.0, 20.0), orientation=(1.0, 0.0, 0.0, 0.0), source_picker='test', confidence=1.0, half_set=HalfSet.A))
        cache = tmp_path / 'cache'
        full, kept_full, _ = extract_particle_set(particles, paths, 11, cache_dir=cache)

        # simulate an abort during tomogram b: its marker never got written
        (cache / 'b.json').unlink()
        (cache / 'b.npy').unlink()
        calls: list[str] = []
        original = extract_module._extract_one_tomogram
        monkeypatch.setattr(extract_module, '_extract_one_tomogram', lambda job: (calls.append(job.tomogram_id), original(job))[1])
        resumed, kept_resumed, _ = extract_particle_set(particles, paths, 11, cache_dir=cache)
        assert calls == ['b']
        assert [p.particle_id for p in kept_resumed] == [p.particle_id for p in kept_full]
        assert np.array_equal(resumed[1], full[1])

        # a different box size invalidates the markers
        calls.clear()
        extract_particle_set(particles, paths, 13, cache_dir=cache)
        assert sorted(calls) == ['a', 'b']

# TestFeatures: class containing unit tests for src/stamp/classify/features.py
class TestFeatures:
    def test_rotational_average_shape(self) -> None:
        '''The rotational average has shape (n_radial_bins, box_voxels).'''
        bin_index, valid = cylindrical_bins(11, n_radial_bins=4)
        averaged = rotational_average(np.ones((11, 11, 11)), bin_index, valid, 4)
        assert averaged.shape == (4, 11)

    def test_features_are_invariant_to_in_plane_rotation(self, tmp_path: Path) -> None:
        '''A subvolume rotated about z should give the same feature vector.'''
        rng = np.random.default_rng(0)
        subvolume = rng.normal(size=(21, 21, 21))
        # Blur slightly so interpolation during rotation doesn't dominate.
        subvolume = np.cumsum(np.cumsum(subvolume, axis=0), axis=1) / 100.0
        rotated = rotate(subvolume, angle=37.0, axes=(0, 1), reshape=False, order=1)
        subvols = _subvols_from_array(tmp_path, np.stack([subvolume, rotated]))
        features = build_feature_matrix(subvols, n_radial_bins=8, max_azimuthal_mode=0)
        correlation = np.corrcoef(features[0], features[1])[0, 1]
        assert correlation > 0.9, f'in-plane rotation changed the features (r={correlation:.3f})'

    def test_features_distinguish_different_structures(self, tmp_path: Path) -> None:
        '''A blob and an empty box give different features.'''
        box = 21
        centre = box // 2
        with_blob = np.zeros((box, box, box))
        with_blob[centre - 1 : centre + 2, centre - 1 : centre + 2, centre + 4 : centre + 7] = 1.0
        without_blob = np.zeros((box, box, box))
        subvols = _subvols_from_array(tmp_path, np.stack([with_blob, without_blob]))
        features = build_feature_matrix(subvols, n_radial_bins=8)
        assert not np.allclose(features[0], features[1])

    def test_empty_input_returns_empty_matrix(self) -> None:
        '''An empty subvolume stack yields an empty feature matrix.'''
        features = build_feature_matrix(np.empty((0, 11, 11, 11)), n_radial_bins=4)
        assert features.shape[0] == 0

    def test_mode_zero_reproduces_rotational_average(self, tmp_path: Path) -> None:
        '''max_azimuthal_mode=0 must give features equivalent to rotational average.'''
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(4, 21, 21, 21))
        subvolumes = np.cumsum(np.cumsum(raw, axis=1), axis=2) / 100.0

        subvols = _subvols_from_array(tmp_path, subvolumes)
        azimuthal = build_feature_matrix(subvols, n_radial_bins=10, max_azimuthal_mode=0)
        bin_index, valid = cylindrical_bins(21, n_radial_bins=10)
        classic = np.stack([rotational_average(v, bin_index, valid, 10).ravel() for v in subvolumes])
        classic = (classic - classic.mean(axis=1, keepdims=True)) / classic.std(axis=1, keepdims=True)

        for row_a, row_b in zip(azimuthal, classic):
            assert np.corrcoef(row_a, row_b)[0, 1] > 0.95

    def test_azimuthal_magnitudes_are_rotation_invariant(self) -> None:
        particle = _c_n_particle(n_fold=4)
        rotated = _rotate_in_plane(particle, 31.0)
        original = azimuthal_magnitudes(particle, 12, 64, 6)
        turned = azimuthal_magnitudes(rotated, 12, 64, 6)
        correlation = np.corrcoef(original.ravel(), turned.ravel())[0, 1]
        assert correlation > 0.95, (f'azimuthal magnitudes changed under in-plane rotation (r={correlation:.3f})')

    def test_c4_particle_has_power_at_mode_four(self) -> None:
        '''A four-fold ring should put conspicuous power in m=4 relative to its neighbouring modes.'''
        magnitudes = azimuthal_magnitudes(_c_n_particle(n_fold=4), 12, 64, 6)
        per_mode = magnitudes.reshape(magnitudes.shape[0], -1).sum(axis=1)
        assert per_mode[4] > per_mode[3]
        assert per_mode[4] > per_mode[5]

    def test_c3_particle_has_power_at_mode_three(self) -> None:
        '''A three-fold ring puts conspicuous power in m=3.'''
        magnitudes = azimuthal_magnitudes(_c_n_particle(n_fold=3), 12, 64, 6)
        per_mode = magnitudes.reshape(magnitudes.shape[0], -1).sum(axis=1)
        assert per_mode[3] > per_mode[2]
        assert per_mode[3] > per_mode[4]

    def test_c3_and_c4_separate_with_azimuthal_features_but_not_without(self, tmp_path: Path) -> None:
        '''Two particles differing only in symmetry are indistinguishable to a rotational average and distinguishable once azimuthal modes are included.'''
        rng = np.random.default_rng(0)
        subvolumes = []
        labels = []
        for _ in range(15):
            for n_fold in (3, 4):
                particle = _c_n_particle(n_fold=n_fold)
                particle = _rotate_in_plane(particle, rng.uniform(0, 360))
                particle = particle + rng.normal(0, 0.05, size=particle.shape)
                subvolumes.append(particle)
                labels.append(n_fold)
        subvolumes = np.stack(subvolumes)
        labels = np.array(labels)

        def _between_group_separation(features: np.ndarray) -> float:
            '''Ratio of between-group to within-group distance (1 means the groups are indistinguishable).'''
            group_three = features[labels == 3]
            group_four = features[labels == 4]
            between = np.linalg.norm(group_three.mean(0) - group_four.mean(0))
            within = 0.5 * (group_three.std(0).mean() + group_four.std(0).mean())
            return between / max(within, 1e-9)

        subvols = _subvols_from_array(tmp_path, subvolumes)
        rotational_only = build_feature_matrix(subvols, n_radial_bins=12, max_azimuthal_mode=0)
        with_azimuthal = build_feature_matrix(subvols, n_radial_bins=12, max_azimuthal_mode=4)

        assert _between_group_separation(with_azimuthal) > (1.5 * _between_group_separation(rotational_only)), 'azimuthal features did not improve C3-vs-C4 separation'

    def test_azimuthal_sampling_must_resolve_requested_mode(self) -> None:
        '''Undersampled FFT raises rather than aliasing.'''
        with pytest.raises(ValueError, match='aliasing'):
            azimuthal_magnitudes(_c_n_particle(), n_radial_bins=12, n_azimuthal_samples=6, max_mode=6)

    def test_inner_bins_excluded_from_azimuthal_block(self, tmp_path: Path) -> None:
        '''Raising min_radius_fraction should shrink the feature vector.'''
        subvolumes = np.stack([_c_n_particle() for _ in range(3)])
        subvols = _subvols_from_array(tmp_path, subvolumes)
        wide = build_feature_matrix(
            subvols, n_radial_bins=12, max_azimuthal_mode=4, min_radius_fraction=0.0
        )
        narrow = build_feature_matrix(
            subvols, n_radial_bins=12, max_azimuthal_mode=4, min_radius_fraction=0.5
        )
        assert narrow.shape[1] < wide.shape[1]

    def test_degenerate_rows_are_all_zero(self, tmp_path: Path):
        '''build_feature_matrix z-scores a flat subvolume to zeros.'''
        stack = np.stack([np.zeros((11, 11, 11)), np.random.default_rng(0).random((11, 11, 11))])
        subvols = _subvols_from_array(tmp_path, stack)
        features = build_feature_matrix(subvols, n_radial_bins=4, max_azimuthal_mode=0)
        assert not features[0].any() and features[1].any()

    def test_degenerate_rows_stay_zero_with_azimuthal_modes(self, tmp_path: Path):
        '''An empty subvolume is all-zero in every block, including column-standardised modes 1+.'''
        rng = np.random.default_rng(0)
        stack = np.concatenate([np.zeros((1, 15, 15, 15)), rng.random((6, 15, 15, 15))])
        for normalisation in ('none', 'particle', 'tomogram', 'both'):
            features = build_feature_matrix(_subvols_from_array(tmp_path / normalisation, stack), n_radial_bins=6, max_azimuthal_mode=3, normalisation=normalisation)
            assert not features[0].any() and features[1:].any(axis=1).all()

    def test_feature_checkpoint_skips_recomputation(self, tmp_path: Path, monkeypatch) -> None:
        from stamp.classify import features as features_module
        subvols = _subvols_from_array(tmp_path / 'sv', np.random.default_rng(0).random((6, 11, 11, 11)))
        cache = tmp_path / 'features'
        first = build_feature_matrix(subvols, n_radial_bins=4, cache_dir=cache)

        def _fail(*args, **kwargs):
            raise AssertionError('features recomputed despite checkpoint')

        monkeypatch.setattr(features_module, 'azimuthal_magnitudes', _fail)
        assert np.array_equal(build_feature_matrix(subvols, n_radial_bins=4, cache_dir=cache), first)

        # a checkpoint built for different rows is ignored
        with pytest.raises(AssertionError):
            build_feature_matrix(subvols.take([0, 1, 2]), n_radial_bins=4, cache_dir=cache)

    def test_particle_normalisation_removes_amplitude_scale(self, tmp_path: Path) -> None:
        '''Amplifying one particle's signal disturbs other particles' features if normalisation isn't applied.'''
        rng = np.random.default_rng(0)
        base = np.stack([_rotate_in_plane(_c_n_particle(n_fold=4), rng.uniform(0, 360)) + rng.normal(0, 0.05, (25, 25, 25)) for _ in range(8)])
        loud = base.copy()
        loud[0] = base[0].mean() + 20.0 * (base[0] - base[0].mean())
        kwargs = dict(n_radial_bins=8, max_azimuthal_mode=3)
        shifts = {}
        for mode in ('none', 'particle'):
            before = build_feature_matrix(_subvols_from_array(tmp_path / f'{mode}_a', base), normalisation=mode, **kwargs)
            after = build_feature_matrix(_subvols_from_array(tmp_path / f'{mode}_b', loud), normalisation=mode, **kwargs)
            shifts[mode] = np.abs(before[1:] - after[1:]).mean()  # the untouched particles
        assert shifts['particle'] < 0.2 * shifts['none']

    def test_tomogram_normalisation_removes_tomogram_offsets(self, tmp_path: Path) -> None:
        '''Two tomograms with different mode 1+ levels are centred on each other post-tomogram standardisation.'''
        rng = np.random.default_rng(0)
        n, box = 60, 15  # n is at least the per-tomogram minimum, so each tomogram is standardised on its own
        tmp_path.mkdir(parents=True, exist_ok=True)
        np.save(tmp_path / 'quiet.npy', rng.normal(0, 0.05, (n, box, box, box)).astype(np.float32))
        np.save(tmp_path / 'loud.npy', rng.normal(0, 0.5, (n, box, box, box)).astype(np.float32))
        subvols = Subvolumes(tmp_path, [('quiet', i) for i in range(n)] + [('loud', i) for i in range(n)], box)
        kwargs = dict(n_radial_bins=6, max_azimuthal_mode=2, min_radius_fraction=0.0)
        mode0_width = 6 * box
        population = build_feature_matrix(subvols, normalisation='none', **kwargs)[:, mode0_width:]
        per_tomogram = build_feature_matrix(subvols, normalisation='tomogram', **kwargs)[:, mode0_width:]
        gap_population = np.abs(population[:n].mean() - population[n:].mean())
        gap_tomogram = np.abs(per_tomogram[:n].mean() - per_tomogram[n:].mean())
        assert gap_tomogram < 0.1 * gap_population

    def test_unknown_normalisation_is_rejected(self, tmp_path: Path) -> None:
        subvols = _subvols_from_array(tmp_path, np.random.default_rng(0).random((4, 11, 11, 11)))
        with pytest.raises(ValueError, match='normalisation'):
            build_feature_matrix(subvols, n_radial_bins=4, normalisation='bogus')

    # test_axial_features_ignore_azimuth_and_axial_shift: rods of two kinds at random azimuth and axial position group by kind with axial_invariant, and by position without it
    def test_axial_features_ignore_azimuth_and_axial_shift(self, tmp_path: Path) -> None:
        box, rng = 25, np.random.default_rng(0)
        rods = [rod_object(box, seed, repeat=box) for seed in (3, 4)]
        labels = np.repeat([0, 1], 15)
        volumes = [roll_about_normal(np.roll(rods[k], int(rng.integers(0, box)), axis=2), float(rng.uniform(0, 360))) for k in labels]
        subvols = _subvols_from_array(tmp_path, np.stack(volumes))
        kwargs = dict(n_radial_bins=6, max_azimuthal_mode=3, n_azimuthal_samples=32, normalisation='both')
        assert _nearest_neighbour_accuracy(build_feature_matrix(subvols, axial_invariant=True, **kwargs), labels) >= 0.95
        assert _nearest_neighbour_accuracy(build_feature_matrix(subvols, **kwargs), labels) < 0.9

# TestFrames: class containing tests for frame and rotation conventions
class TestFrames:
    # test_rotate_density_moves_features_by_the_matrix: a feature at +x moves to +y under a +90 degree azimuth and to -z under the flip
    def test_rotate_density_moves_features_by_the_matrix(self) -> None:
        volume = _blob((15, 15, 15), (10, 7, 9))    # offset (+3, 0, +2) from the centre
        assert _peak(rotate_density(volume, azimuth_matrix(90.0))) == (7, 10, 9)
        assert _peak(rotate_density(volume, flip_matrix())) == (10, 7, 5)

    # test_roll_about_normal_is_azimuth_matrix: the existing scipy roll equals rotate_density with azimuth_matrix for random angles
    def test_roll_about_normal_is_azimuth_matrix(self) -> None:
        rng = np.random.default_rng(0)
        volume = gaussian_filter(rng.normal(size=(17, 17, 17)).astype(np.float32), 2.0)
        for angle in rng.uniform(-180, 180, 5):
            rolled = roll_about_normal(volume, float(angle))
            moved = rotate_density(volume, azimuth_matrix(float(angle)))
            inside = np.s_[4:-4, 4:-4, 4:-4]
            assert np.allclose(rolled[inside], moved[inside], atol=1e-3)

    # test_azimuth_degrees_inverts_azimuth_matrix: angle round trip across the full range
    def test_azimuth_degrees_inverts_azimuth_matrix(self) -> None:
        for angle in (-170.0, -45.0, 0.0, 30.0, 179.0):
            assert np.isclose(azimuth_degrees(azimuth_matrix(angle)), angle)

    # test_rotation_to_quaternion_round_trip: quaternion_to_matrix recovers the matrix, w is non-negative
    def test_rotation_to_quaternion_round_trip(self) -> None:
        rotation = azimuth_matrix(70.0) @ flip_matrix()
        quaternion = rotation_to_quaternion(rotation)
        assert quaternion[0] >= 0 and np.allclose(quaternion_to_matrix(quaternion), rotation)

    # test_off_axis_density_lands_on_the_normal: a density planted 8 voxels along a tilted normal sits on +z of the extracted box
    def test_off_axis_density_lands_on_the_normal(self) -> None:
        normal = np.array([0.6, 0.0, 0.8])
        position = np.array([30.0, 30.0, 30.0])
        tomogram = np.zeros((61, 61, 61), dtype=np.float32)    # zyx array
        target_xyz = position + 8.0 * normal
        tomogram[int(round(target_xyz[2])), int(round(target_xyz[1])), int(round(target_xyz[0]))] = 1.0
        tomogram = gaussian_filter(tomogram, 1.0)
        quaternion = quaternion_from_reference_to(normal)
        box = extract_subvolume(tomogram, tuple(position), quaternion, 21)
        assert _peak(box) == (10, 10, 18)   # centre 10, plus 8 along +z

# TestMatch: class containing unit tests for prior-aware particle alignment to reference
class TestMatch:
    # test_rank_rotations_matches_masked_bank: the shared-bank score equals the brute-force score against a bank masked with each particle's own wedge
    def test_rank_rotations_matches_masked_bank(self) -> None:
        rng = np.random.default_rng(0)
        reference = asymmetric_object(_BOX, 1)
        space = search_space(OrientationPrior.NORMAL, _BOX, angular_step_degrees=60.0)
        bank = rotated_bank(reference, space.rotations)
        orientations = [_random_orientation(rng) for _ in range(3)]
        wedges = np.stack([box_wedge(_BOX, o, 60.0, 90.0) for o in orientations])
        particles = np.stack([apply_wedge(rng.normal(size=(_BOX,) * 3), w) for w in wedges])
        fast, _norms = rank_rotations(wedge_filter(particles, wedges), wedges, bank, _BOX)
        for i, wedge in enumerate(wedges):
            for r, rotated in enumerate(bank):
                masked = apply_wedge(rotated.reshape((_BOX,) * 3), wedge).ravel()
                expected = (particles[i].ravel() @ masked) / (np.linalg.norm(particles[i]) * np.linalg.norm(masked))
                assert abs(fast[i, r] - expected) < 1e-3

    # test_normal_prior_recovers_azimuth: azimuth error against known rotation, with a different wedge per particle and small shifts
    def test_normal_prior_recovers_azimuth(self) -> None:
        rng = np.random.default_rng(3)
        reference = asymmetric_object(_BOX, 5)
        space = search_space(OrientationPrior.NORMAL, _BOX, angular_step_degrees=10.0, shift_range_voxels=2)
        truth = rng.uniform(0, 360, 40)
        shifts = rng.integers(-1, 2, (40, 3))
        wedges = np.stack([box_wedge(_BOX, _random_orientation(rng), 60.0, 90.0) for _ in truth])
        particles = np.stack([make_particle(reference, azimuth_matrix(a), tuple(s), w, 0.05, rng) for a, s, w in zip(truth, shifts, wedges)])
        result = align_to_reference(particles, wedges, reference, space)
        found = np.array([azimuth_degrees(space.rotations[i]) for i in result.rotation_index])
        error = _angle_error(found, truth)
        assert np.median(error) <= 5.0 and np.percentile(error, 90) <= 10.0
        assert (result.shift == shifts).all(axis=1).mean() > 0.9

    # test_axis_prior_recovers_azimuth_polarity_and_shift: rods with random polarity and axial offsets
    def test_axis_prior_recovers_azimuth_polarity_and_shift(self) -> None:
        rng = np.random.default_rng(4)
        reference = rod_object(_BOX, 6, repeat=_BOX)    # no helical degeneracy inside the box
        space = search_space(OrientationPrior.AXIS, _BOX, angular_step_degrees=15.0)
        n = 30
        azimuth = rng.uniform(0, 360, n)
        flipped = rng.random(n) < 0.5
        axial = rng.integers(-5, 6, n)
        wedges = np.stack([box_wedge(_BOX, _random_orientation(rng), 60.0, 90.0) for _ in range(n)])
        particles = np.stack([
            make_particle(reference, azimuth_matrix(a) @ (flip_matrix() if f else np.eye(3)), (0, 0, z), w, 0.05, rng)
            for a, f, z, w in zip(azimuth, flipped, axial, wedges)
        ])
        result = align_to_reference(particles, wedges, reference, space)
        found_flip = space.polarity[result.rotation_index] < 0
        assert (found_flip == flipped).mean() >= 0.95
        assert (np.abs(result.shift[:, 2] - axial) <= 1).mean() >= 0.9
        found_azimuth = np.array([azimuth_degrees(space.rotations[i] @ (flip_matrix() if f else np.eye(3))) for i, f in zip(result.rotation_index, found_flip)])
        assert np.median(_angle_error(found_azimuth, azimuth)) <= 8.0

    # test_none_prior_recovers_bank_rotation: a wedged rotated copy finds its own rotation in a noiseless bank (regression for the original brute-force search)
    def test_none_prior_recovers_bank_rotation(self) -> None:
        reference = asymmetric_object(_BOX, 7)
        space = search_space(OrientationPrior.NONE, _BOX, n_rotations=200, shift_range_voxels=1)
        wedge = box_wedge(_BOX, None, 60.0, 90.0)
        target = 123
        particle = apply_wedge(rotate_density(reference, space.rotations[target]), wedge)
        result = align_to_reference(particle[None], wedge[None], reference, space, top=5)
        assert result.rotation_index[0] == target and result.cc[0] > 0.95 and not result.shift.any()

# TestMerge: class containing unit tests for half-split ceilings, merging and phase-randomised null
class TestMerge:
    # test_pair_threshold_ignores_inconsistent_reference: a junk reference with a low ceiling cannot lower the merge bar for real ones
    def test_pair_threshold_ignores_inconsistent_reference(self) -> None:
        similarity = np.array([[1.0, 0.5, 0.4], [0.5, 1.0, 0.4], [0.4, 0.4, 1.0]])
        assert len(np.unique(merge_groups(similarity, pair_thresholds(np.array([0.9, 0.9, 0.2]), 1.0)))) == 3

    # test_merge_groups_joins_chains: similarity at or above the threshold links references transitively
    def test_merge_groups_joins_chains(self) -> None:
        similarity = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.9], [0.1, 0.9, 1.0]])
        assert len(np.unique(merge_groups(similarity, 0.8))) == 1

    # test_rotated_copy_is_the_same_reference: a reference and a rotated, shifted copy are maximally similar and the pose that maps one onto the other is returned
    def test_rotated_copy_is_the_same_reference(self) -> None:
        from stamp.classify.frames import rotate_density
        space = search_space(OrientationPrior.NORMAL, _REF_BOX, angular_step_degrees=15.0, shift_range_voxels=2)
        reference = asymmetric_object(_REF_BOX, 1)
        copy = rotate_density(reference, space.rotations[5])
        similarity, poses = pairwise_similarity([reference, copy, asymmetric_object(_REF_BOX, 2)], space)
        assert similarity[0, 1] > 0.98 and similarity[0, 2] < 0.8
        assert np.allclose(poses[0][1][0], space.rotations[5])

    # test_ceiling_drops_when_poses_are_wrong: half-split agreement is high for a label with correct poses and lower when the poses are scrambled
    def test_ceiling_drops_when_poses_are_wrong(self, tmp_path: Path) -> None:
        data = synthetic_set(tmp_path, [asymmetric_object(_REF_BOX, 1)], OrientationPrior.NORMAL, 40, seed=4)
        space = search_space(OrientationPrior.NORMAL, _REF_BOX, angular_step_degrees=15.0, shift_range_voxels=2)
        labels = np.zeros(len(data.labels), dtype=int)
        scrambled = data.rotations[np.random.default_rng(0).permutation(len(labels))]
        right = half_split_ceilings(data.store.batches(), labels, data.rotations, data.shifts, space, 1, _REF_BOX)[0]
        wrong = half_split_ceilings(data.store.batches(), labels, scrambled, data.shifts, space, 1, _REF_BOX)[0]
        assert right > 0.8 and wrong < right

    # test_null_is_below_real_signal: real particles beat the rejection threshold and the agreement bar is below a real pure label's ceiling
    def test_null_is_below_real_signal(self, tmp_path: Path) -> None:
        objects = [asymmetric_object(_REF_BOX, 1)]
        data = synthetic_set(tmp_path, objects, OrientationPrior.NORMAL, 40, noise_sigma=0.1, seed=5)
        space = search_space(OrientationPrior.NORMAL, _REF_BOX, angular_step_degrees=15.0, shift_range_voxels=2)
        threshold = null_threshold(data.store, objects, space, quantile=0.95)
        assignment = hard_assign(score_against_refs(data.store, objects, space), space)
        assert (assignment.cc > threshold).mean() > 0.9
        bar = null_bar(data.store, objects[0], space, 30, sigmas=3.0)
        ceiling = half_split_ceilings(data.store.batches(), assignment.labels, assignment.rotations, assignment.shifts, space, 1, _REF_BOX)[0]
        assert ceiling > bar

# TestPolish: class containing unit tests for polish step
class TestPolish:
    # test_polish_repairs_corrupted_seeds: noisy seeds are polished into pure classes, for every prior
    @pytest.mark.parametrize('prior', [OrientationPrior.NORMAL, OrientationPrior.AXIS, OrientationPrior.NONE])
    def test_polish_repairs_corrupted_seeds(self, tmp_path: Path, prior: OrientationPrior) -> None:
        make = rod_object if prior is OrientationPrior.AXIS else asymmetric_object
        objects = [make(_BOX, seed) if prior is not OrientationPrior.AXIS else rod_object(_BOX, seed, repeat=_BOX) for seed in (1, 2)]
        data = synthetic_set(tmp_path / 'data', objects, prior, 30, noise_sigma=0.1, seed=3)
        seeds, embedding = _corrupted_seeds(data.labels, 0.15, 4)
        result = run_polish(data.store, embedding, seeds, _config(prior))
        assert _purity(result.labels, data.labels) > 0.9
        assert (result.labels >= 0).mean() > 0.7
        assert len(result.references) == 2

    # test_polish_rejects_noise_only_particles: pure noise matches no reference above the null
    def test_polish_rejects_noise_only_particles(self, tmp_path: Path) -> None:
        objects = [asymmetric_object(_BOX, 1)]
        data = synthetic_set(tmp_path / 'data', objects, OrientationPrior.NORMAL, 40, seed=5)
        noise = synthetic_set(tmp_path / 'noise', [np.zeros_like(objects[0])], OrientationPrior.NORMAL, 20, noise_sigma=0.5, seed=6)
        # one store holding both: build by stacking the volumes of the two sets
        both = _concatenate(tmp_path / 'both', data, noise)
        seeds = np.concatenate([np.zeros(40, dtype=int), np.full(20, -1)])
        embedding = np.zeros((60, 2))
        result = run_polish(both, embedding, seeds, _config(OrientationPrior.NORMAL))
        assert (result.labels[40:] == -1).mean() > 0.7 and (result.labels[:40] >= 0).mean() > 0.8

    # test_polish_resumes_identically: a run stopped after the seeds and restarted gives identical output
    def test_polish_resumes_identically(self, tmp_path: Path) -> None:
        objects = [asymmetric_object(_BOX, 1), asymmetric_object(_BOX, 2)]
        data = synthetic_set(tmp_path / 'data', objects, OrientationPrior.NORMAL, 25, seed=7)
        seeds, embedding = _corrupted_seeds(data.labels, 0.1, 8)
        config = _config(OrientationPrior.NORMAL)
        fresh = run_polish(data.store, embedding, seeds, config, checkpoint_dir=tmp_path / 'ckpt')
        for path in (tmp_path / 'ckpt').glob('round*.npz'):
            path.unlink()       # as if killed between the seed references and the first round
        resumed = run_polish(data.store, embedding, seeds, config, checkpoint_dir=tmp_path / 'ckpt')
        assert np.array_equal(fresh.labels, resumed.labels) and np.allclose(fresh.rotations, resumed.rotations)

    # test_polish_needs_a_seed: no seed cluster of the minimum size is an error
    def test_polish_needs_a_seed(self, tmp_path: Path) -> None:
        from stamp.utils.errors import StampPipelineError
        data = synthetic_set(tmp_path / 'data', [asymmetric_object(_BOX, 1)], OrientationPrior.NORMAL, 12, seed=9)
        with pytest.raises(StampPipelineError):
            run_polish(data.store, np.zeros((12, 2)), np.full(12, -1), _config(OrientationPrior.NORMAL))

# TestPrepare: class containing tests for subvolume preparation
class TestPrepare:
    # test_working_box_is_odd: odd boxes bin to odd boxes so the centre stays on a voxel
    def test_working_box_is_odd(self) -> None:
        assert [working_box(n, 2) for n in (45, 47, 61)] == [23, 23, 31]
        assert working_box(45, 1) == 45

    # test_normalise_uses_masked_statistics: zero mean and unit variance inside the mask
    def test_normalise_uses_masked_statistics(self) -> None:
        rng = np.random.default_rng(0)
        mask = soft_mask(21, 8.0)
        out = normalise(rng.normal(5.0, 3.0, (21,) * 3).astype(np.float32), mask)
        assert abs(out[mask > 0.5].mean()) < 1e-4 and abs(out[mask > 0.5].std() - 1.0) < 1e-3

    # test_fourier_bin_preserves_a_smooth_object: a low-pass object keeps its shape after binning, checked at the centre voxel
    def test_fourier_bin_preserves_a_smooth_object(self) -> None:
        volume = asymmetric_object(45, 0, n_blobs=1, spread=0.0)
        binned = fourier_bin(volume, 23)
        assert binned.shape == (23,) * 3 and binned[11, 11, 11] > 0.9 * volume[22, 22, 22]

    # test_prepare_particle_shape: output is the working box
    def test_prepare_particle_shape(self) -> None:
        assert prepare_particle(np.random.default_rng(1).normal(size=(45,) * 3).astype(np.float32), 23).shape == (23,) * 3

# TestReferences: class containing unit tests for particle store and reference scoring
class TestReferences:
    # test_store_serves_wedges_and_batches: batches cover every particle once and carry each particle's own wedge
    def test_store_serves_wedges_and_batches(self, tmp_path: Path) -> None:
        data = synthetic_set(tmp_path, [asymmetric_object(_REF_BOX, 1)], OrientationPrior.NORMAL, 12, seed=1)
        seen = []
        for indices, volumes, wedges in data.store.batches(5):
            assert volumes.shape == (len(indices), _REF_BOX, _REF_BOX, _REF_BOX) and wedges.shape == volumes.shape
            assert np.array_equal(wedges[0], box_wedge(_REF_BOX, data.store.orientations[indices[0]], 60.0, 90.0))
            seen += indices
        assert sorted(seen) == list(range(12))

    # test_build_store_bins_and_normalises: the prepared store has the working box and matches prepare_particle
    def test_build_store_bins_and_normalises(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(4, 45, 45, 45)).astype(np.float32)
        np.save(tmp_path / 't0.npy', raw)
        store = build_store(Subvolumes(tmp_path, [('t0', i) for i in range(4)], 45), [None] * 4, 23, 60.0, 90.0, tmp_path / 'prepared')
        assert store.volumes.box_voxels == 23 and len(store) == 4
        assert np.allclose(store.volumes[2], prepare_particle(raw[2], 23), atol=1e-5)

    # test_scoring_against_true_objects_recovers_classes: assignment against the generating objects labels particles correctly
    def test_scoring_against_true_objects_recovers_classes(self, tmp_path: Path) -> None:
        objects = [asymmetric_object(_REF_BOX, 1), asymmetric_object(_REF_BOX, 2)]
        data = synthetic_set(tmp_path, objects, OrientationPrior.NORMAL, 20, seed=2)
        space = search_space(OrientationPrior.NORMAL, _REF_BOX, angular_step_degrees=15.0, shift_range_voxels=2)
        assignment = hard_assign(score_against_refs(data.store, objects, space), space)
        assert (assignment.labels == data.labels).mean() > 0.95
        assert (assignment.margin > 0).all()

    # test_rebuild_from_true_poses_matches_object: references rebuilt from the true labels and poses correlate with the generating object
    def test_rebuild_from_true_poses_matches_object(self, tmp_path: Path) -> None:
        objects = [asymmetric_object(_REF_BOX, 1), asymmetric_object(_REF_BOX, 2)]
        data = synthetic_set(tmp_path, objects, OrientationPrior.NONE, 25, noise_sigma=0.05, seed=3)
        rebuilt = rebuild_references(data.store, data.labels, data.rotations, data.shifts, 2)
        for reference, truth in zip(rebuilt, objects):
            assert float((reference * truth).sum() / np.linalg.norm(reference) / np.linalg.norm(truth)) > 0.9

# TestSearch: class containing unit tests for per-prior alignment search space
class TestSearch:
    # test_normal_space_is_z_rolls: 36 proper rotations about z at a 10 degree step, no polarity flips
    def test_normal_space_is_z_rolls(self) -> None:
        space = search_space(OrientationPrior.NORMAL, 25, angular_step_degrees=10.0)
        assert len(space) == 36 and (space.polarity == 1).all()
        assert np.allclose(space.rotations[:, 2], [0.0, 0.0, 1.0]) and np.allclose(space.rotations[:, :, 2], [0.0, 0.0, 1.0])
        assert np.allclose(np.linalg.det(space.rotations), 1.0)

    # test_axis_space_doubles_with_polarity: z-rolls and their flips, with an axial shift window of a quarter box
    def test_axis_space_doubles_with_polarity(self) -> None:
        space = search_space(OrientationPrior.AXIS, 25, angular_step_degrees=10.0)
        assert len(space) == 72 and (space.polarity == -1).sum() == 36
        assert np.allclose(space.rotations[space.polarity < 0][:, 2, 2], -1.0)
        assert space.max_shift == (3, 3, 6)

    # test_none_space_starts_at_identity: identity first, all proper rotations, reproducible from the seed
    def test_none_space_starts_at_identity(self) -> None:
        space = search_space(OrientationPrior.NONE, 25, n_rotations=50, seed=4)
        assert len(space) == 50 and np.allclose(space.rotations[0], np.eye(3))
        assert np.allclose(np.linalg.det(space.rotations), 1.0)
        assert np.array_equal(space.rotations, search_space(OrientationPrior.NONE, 25, n_rotations=50, seed=4).rotations)

    # test_rotated_bank_matches_rotate_density: threaded and serial banks agree with rotating one by one
    def test_rotated_bank_matches_rotate_density(self) -> None:
        reference = asymmetric_object(15, 0)
        rotations = search_space(OrientationPrior.NONE, 15, n_rotations=6).rotations
        serial = rotated_bank(reference, rotations)
        assert np.allclose(serial, rotated_bank(reference, rotations, threads=3))
        assert np.allclose(serial[3], rotate_density(reference, rotations[3]).ravel())

# TestWedge: class containing tests for per-particle missing wedge
class TestWedge:
    # test_identity_orientation_is_the_global_mask: tilt about y with identity orientation keeps |kz| <= |kx| tan(tilt)
    def test_identity_orientation_is_the_global_mask(self) -> None:
        box = 15
        freq = np.fft.fftfreq(box)
        kx, _ky, kz = np.meshgrid(freq, freq, freq, indexing='ij')
        expected = (np.abs(kz) <= np.abs(kx) * np.tan(np.radians(60.0))).astype(np.float32)
        assert np.array_equal(box_wedge(box, None, 60.0, 90.0), expected)

    # test_rotated_wedge_matches_brute_force: random orientations and tilt axes agree with the explicit construction
    def test_rotated_wedge_matches_brute_force(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(3):
            normal = rng.normal(size=3)
            orientation = quaternion_from_reference_to(normal)
            tilt_axis = float(rng.uniform(0, 180))
            fast = box_wedge(9, orientation, 55.0, tilt_axis)
            slow = _brute_wedge(9, orientation, 55.0, tilt_axis)
            assert (fast != slow).mean() < 0.01     # only voxels exactly on the edge may differ by rounding

    # test_wedge_is_centrosymmetric: the mask is symmetric under k -> -k so filtered volumes stay real
    def test_wedge_is_centrosymmetric(self) -> None:
        orientation = quaternion_from_reference_to(np.array([0.3, -0.5, 0.8]))
        wedge = box_wedge(11, orientation, 50.0, 30.0)
        assert np.array_equal(wedge, np.roll(wedge[::-1, ::-1, ::-1], 1, axis=(0, 1, 2)))

    # test_no_tilt_means_no_wedge: unset tilt keeps everything
    def test_no_tilt_means_no_wedge(self) -> None:
        assert box_wedge(7, None, None, 90.0).all()
        assert observed_directions(np.eye(3), None, None, 90.0).all()

    # test_apply_wedge_is_idempotent: filtering twice changes nothing
    def test_apply_wedge_is_idempotent(self) -> None:
        rng = np.random.default_rng(2)
        wedge = box_wedge(11, None, 60.0, 90.0)
        once = apply_wedge(rng.normal(size=(11, 11, 11)), wedge)
        assert np.allclose(apply_wedge(once, wedge), once, atol=1e-5)
