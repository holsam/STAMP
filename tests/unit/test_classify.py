'''
STAMP: unit tests for subvolume extraction, feature classification, and clustering
'''

# Import external dependencies
import mrcfile, numpy as np, pytest
from scipy.ndimage import rotate

# Import internal STAMP objects
from stamp.classify.cluster import (
    ClusteringConfig,
    label_to_cluster_id,
    match_clusters_across_halves,
    reduce_and_cluster,
    reduce_and_cluster_shared,
)
from stamp.classify.extract import (
    box_fits_inside,
    canonical_grid,
    extract_particle_set,
    extract_subvolume,
    quaternion_to_matrix,
)
from stamp.classify.features import azimuthal_magnitudes, build_feature_matrix, cylindrical_bins, rotational_average
from stamp.picking.geometry import quaternion_from_reference_to
from stamp.schemas.particles import HalfSet, Particle

# _three_blobs: three well-separated Gaussian blobs in 2D
def _three_blobs(n_per_group: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
    return np.vstack(
        [rng.normal(centre, 1.0, size=(n_per_group, 2)) for centre in centres]
    )

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

# _rotate_in_plane: rotate a canonical (x, y, z) volume in-plane about axes (0, 1)
def _rotate_in_plane(volume: np.ndarray, degrees: float) -> np.ndarray:
    return rotate(volume, angle=degrees, axes=(0, 1), reshape=False, order=1)

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

        particle = Particle(particle_id='p0', tomogram_id='t', position=(20.0, 20.0, 20.0), orientation=None, source_picker='test', confidence=1.0, half_set=HalfSet.A)
        subvolumes, kept, _ = extract_particle_set([particle], {'t': str(tmp_path / 't.mrc')}, 11, segmentation_paths={'t': str(tmp_path / 's.mrc')})
        assert len(kept) == 1
        # the box centre sampled the membrane plane; it must now sit near background
        assert abs(float(subvolumes[0, 5, 5, 5])) < 5.0

    def test_no_segmentation_leaves_tomogram_untouched(self, tmp_path) -> None:
        shape = (40, 40, 40)
        tomogram = np.zeros(shape, dtype=np.float32)
        tomogram[:, :, 20] = 50.0
        with mrcfile.new(tmp_path / 't.mrc', overwrite=True) as mrc:
            mrc.set_data(tomogram)
            mrc.voxel_size = 1.0
        particle = Particle(particle_id='p0', tomogram_id='t', position=(20.0, 20.0, 20.0), orientation=None, source_picker='test', confidence=1.0, half_set=HalfSet.A)
        subvolumes, kept, _ = extract_particle_set([particle], {'t': str(tmp_path / 't.mrc')}, 11)
        assert float(subvolumes[0, 5, 5, 5]) == 50.0

# TestFeatures: class containing unit tests for src/stamp/classify/features.py
class TestFeatures:
    def test_rotational_average_shape(self) -> None:
        '''The rotational average has shape (n_radial_bins, box_voxels).'''
        bin_index, valid = cylindrical_bins(11, n_radial_bins=4)
        averaged = rotational_average(np.ones((11, 11, 11)), bin_index, valid, 4)
        assert averaged.shape == (4, 11)

    def test_features_are_invariant_to_in_plane_rotation(self) -> None:
        '''A subvolume rotated about z should give the same feature vector.'''
        rng = np.random.default_rng(0)
        subvolume = rng.normal(size=(21, 21, 21))
        # Blur slightly so interpolation during rotation doesn't dominate.
        subvolume = np.cumsum(np.cumsum(subvolume, axis=0), axis=1) / 100.0
        rotated = rotate(subvolume, angle=37.0, axes=(0, 1), reshape=False, order=1)
        features = build_feature_matrix(np.stack([subvolume, rotated]), n_radial_bins=8, max_azimuthal_mode=0)
        correlation = np.corrcoef(features[0], features[1])[0, 1]
        assert correlation > 0.9, f'in-plane rotation changed the features (r={correlation:.3f})'

    def test_features_distinguish_different_structures(self) -> None:
        '''A blob and an empty box give different features.'''
        box = 21
        centre = box // 2
        with_blob = np.zeros((box, box, box))
        with_blob[centre - 1 : centre + 2, centre - 1 : centre + 2, centre + 4 : centre + 7] = 1.0
        without_blob = np.zeros((box, box, box))
        features = build_feature_matrix(np.stack([with_blob, without_blob]), n_radial_bins=8)
        assert not np.allclose(features[0], features[1])

    def test_empty_input_returns_empty_matrix(self) -> None:
        '''An empty subvolume stack yields an empty feature matrix.'''
        features = build_feature_matrix(np.empty((0, 11, 11, 11)), n_radial_bins=4)
        assert features.shape[0] == 0

    def test_mode_zero_reproduces_rotational_average(self) -> None:
        '''max_azimuthal_mode=0 must give features equivalent to rotational average.'''
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(4, 21, 21, 21))
        subvolumes = np.cumsum(np.cumsum(raw, axis=1), axis=2) / 100.0

        azimuthal = build_feature_matrix(subvolumes, n_radial_bins=10, max_azimuthal_mode=0)
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

    def test_c3_and_c4_separate_with_azimuthal_features_but_not_without(self) -> None:
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

        rotational_only = build_feature_matrix(subvolumes, n_radial_bins=12, max_azimuthal_mode=0)
        with_azimuthal = build_feature_matrix(subvolumes, n_radial_bins=12, max_azimuthal_mode=4)

        assert _between_group_separation(with_azimuthal) > (1.5 * _between_group_separation(rotational_only)), 'azimuthal features did not improve C3-vs-C4 separation'

    def test_azimuthal_sampling_must_resolve_requested_mode(self) -> None:
        '''Undersampled FFT raises rather than aliasing.'''
        with pytest.raises(ValueError, match='aliasing'):
            azimuthal_magnitudes(_c_n_particle(), n_radial_bins=12, n_azimuthal_samples=6, max_mode=6)

    def test_inner_bins_excluded_from_azimuthal_block(self) -> None:
        '''Raising min_radius_fraction should shrink the feature vector.'''
        subvolumes = np.stack([_c_n_particle() for _ in range(3)])
        wide = build_feature_matrix(
            subvolumes, n_radial_bins=12, max_azimuthal_mode=4, min_radius_fraction=0.0
        )
        narrow = build_feature_matrix(
            subvolumes, n_radial_bins=12, max_azimuthal_mode=4, min_radius_fraction=0.5
        )
        assert narrow.shape[1] < wide.shape[1]

    def test_degenerate_rows_are_all_zero(self):
        '''build_feature_matrix z-scores a flat subvolume to zeros.'''
        stack = np.stack([np.zeros((11, 11, 11)), np.random.default_rng(0).random((11, 11, 11))])
        features = build_feature_matrix(stack, n_radial_bins=4, max_azimuthal_mode=0)
        assert not features[0].any() and features[1].any()

# TestCluster: class containing unit tests for test_cluster.py
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
        result = reduce_and_cluster(
            features, ClusteringConfig(method='kmeans', n_components=50, n_clusters=2)
        )
        assert result.embedding.shape[1] <= 4

    def test_too_few_particles_raises(self) -> None:
        '''A single particle raises a clear error.'''
        with pytest.raises(ValueError, match='Too few particles'):
            reduce_and_cluster(
                np.zeros((1, 10)), ClusteringConfig(method='kmeans', n_clusters=1)
            )

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

    def test_cluster_matching_pairs_nearest_centroids(self) -> None:
        '''Cross-half matching pairs the nearest centroids.'''
        centroids_a = {0: np.array([0.0, 0.0]), 1: np.array([10.0, 0.0])}
        centroids_b = {0: np.array([10.2, 0.1]), 1: np.array([0.1, 0.2])}
        matches = match_clusters_across_halves(centroids_a, centroids_b)
        assert matches[0][0] == 1
        assert matches[1][0] == 0

    def test_cluster_matching_uses_each_a_cluster_once(self) -> None:
        '''A half-A cluster is matched at most once.'''
        centroids_a = {0: np.array([0.0, 0.0])}
        centroids_b = {0: np.array([0.1, 0.0]), 1: np.array([0.2, 0.0])}
        matches = match_clusters_across_halves(centroids_a, centroids_b)
        assert len([m for m in matches.values() if m[0] == 0]) == 1

    def test_shared_basis_matches_recover_planted_split(self):
        '''Two motifs, each split A/B, match A<->B correctly.'''
        rng = np.random.default_rng(0)
        motif_a = rng.normal(0.0, 1.0, 40)
        motif_b = rng.normal(5.0, 1.0, 40)
        rows, group_a, group_b = [], [], []
        for i in range(40):
            rows.append(motif_a + rng.normal(0, 0.1, 40)); (group_a if i % 2 else group_b).append(len(rows) - 1)
        for i in range(40):
            rows.append(motif_b + rng.normal(0, 0.1, 40)); (group_a if i % 2 else group_b).append(len(rows) - 1)
        features = np.array(rows)
        config = ClusteringConfig(method='kmeans', n_clusters=2, n_components=5)
        results = reduce_and_cluster_shared(features, {'A': np.array(group_a), 'B': np.array(group_b)}, config)
        matches = match_clusters_across_halves(results['A'].centroids, results['B'].centroids)
        assert len(matches) == 2 and all(d < 1.0 for _, d in matches.values())
