'''
STAMP: dimensionality reduction and clustering of particle features
'''

# Import external dependencies
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA

# NOISE_CLUSTER_ID: id for noise cluster
NOISE_CLUSTER_ID = 'noise'

# ClusteringConfig: class containing parameters for dimensionality reduction and clustering run
@dataclass(frozen=True)
class ClusteringConfig:
    method: str = 'hdbscan' # 'hdbscan' or 'kmeans'
    n_components: int = 20  # PCA components
    min_cluster_size: int = 20  # HDBSCAN
    n_clusters: int = 5 # KMeans
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.method not in ('hdbscan', 'kmeans'):
            raise ValueError("method must be 'hdbscan' or 'kmeans'.")

# ClusteringResult: class containing labels, PCA embedding and centroids from a clustering
@dataclass
class ClusteringResult:
    labels: np.ndarray  # -1 means noise (HDBSCAN only)
    embedding: np.ndarray   # PCA coordinates
    explained_variance_ratio: np.ndarray
    centroids: dict[int, np.ndarray]    # label -> centroid in PCA space

# _cluster_embedding: run the configured clusterer on an embedding, return (labels, centroids)
def _cluster_embedding(embedding: np.ndarray, config: ClusteringConfig) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    if config.method == 'hdbscan':
        if embedding.shape[0] < config.min_cluster_size:
            raise ValueError(f'only {embedding.shape[0]} particles but min_cluster_size is {config.min_cluster_size}')
        labels = HDBSCAN(min_cluster_size=config.min_cluster_size).fit_predict(embedding)
    else:
        n_clusters = min(config.n_clusters, embedding.shape[0])
        labels = KMeans(n_clusters=n_clusters, random_state=config.random_state, n_init=10).fit_predict(embedding)
    centroids = {int(label): embedding[labels == label].mean(axis=0) for label in np.unique(labels) if label != -1}
    return labels, centroids

# reduce_and_cluster: PCA followed by HDBSCAN or KMeans
def reduce_and_cluster(features: np.ndarray, config: ClusteringConfig) -> ClusteringResult:
    if features.shape[0] == 0:
        raise ValueError('cannot cluster an empty feature matrix')
    n_components = min(config.n_components, features.shape[0] - 1, features.shape[1])
    if n_components < 2:
        raise ValueError(f'Too few particles ({features.shape[0]}) to run PCA (at least 2 needed)')

    pca = PCA(n_components=n_components, random_state=config.random_state)
    embedding = pca.fit_transform(features)
    labels, centroids = _cluster_embedding(embedding, config)
    return ClusteringResult(
        labels=labels,
        embedding=embedding,
        explained_variance_ratio=pca.explained_variance_ratio_,
        centroids=centroids,
    )

# reduce_and_cluster_shared: one PCA basis fitted on every particle, each index group clustered in that basis
def reduce_and_cluster_shared(features: np.ndarray, groups: dict[str, np.ndarray], config: ClusteringConfig) -> dict[str, ClusteringResult]:
    n_components = min(config.n_components, features.shape[0] - 1, features.shape[1])
    if n_components < 1:
        raise ValueError(f'too few particles ({features.shape[0]}) to run PCA (at least 2 needed)')
    pca = PCA(n_components=n_components, random_state=config.random_state)
    embedding_all = pca.fit_transform(features)
    results: dict[str, ClusteringResult] = {}
    for name, indices in groups.items():
        embedding = embedding_all[indices]
        labels, centroids = _cluster_embedding(embedding, config)
        results[name] = ClusteringResult(
            labels=labels,
            embedding=embedding,
            explained_variance_ratio=pca.explained_variance_ratio_,
            centroids=centroids,
        )
    return results

# label_to_cluster_id: turn a raw cluster label into a stable string id
def label_to_cluster_id(label: int) -> str:
    return NOISE_CLUSTER_ID if label == -1 else f'c{label:02d}'

# match_clusters_across_halves: pair half-B clusters to half-A clusters by nearest centroid
def match_clusters_across_halves(
    centroids_a: dict[int, np.ndarray], centroids_b: dict[int, np.ndarray]
) -> dict[int, tuple[int, float]]:
    matches: dict[int, tuple[int, float]] = {}
    available = set(centroids_a)
    pairs = sorted(
        (
            (float(np.linalg.norm(centroid_b - centroids_a[label_a])), label_b, label_a)
            for label_b, centroid_b in centroids_b.items()
            for label_a in centroids_a
        )
    )
    for distance, label_b, label_a in pairs:
        if label_b in matches or label_a not in available:
            continue
        matches[label_b] = (label_a, distance)
        available.discard(label_a)
    return matches
