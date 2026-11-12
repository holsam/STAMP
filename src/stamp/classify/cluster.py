'''
STAMP: dimensionality reduction and clustering of particle features
'''

# Import external dependencies
import igraph, leidenalg, numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

# Import internal STAMP objects
from stamp.utils.log import log, log_config

# NOISE_CLUSTER_ID: id for noise cluster
NOISE_CLUSTER_ID = 'noise'

# ClusteringConfig: class containing parameters for dimensionality reduction and clustering run
@dataclass(frozen=True)
class ClusteringConfig:
    method: str = 'leiden' # 'leiden', 'hdbscan' or 'kmeans'
    n_components: int = 20  # PCA components
    min_cluster_size: int = 20  # HDBSCAN, and the smallest Leiden community kept
    n_clusters: int = 5 # KMeans
    n_neighbors: int = 15   # Leiden kNN graph
    resolution: float = 1.0 # Leiden modularity resolution
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.method not in ('hdbscan', 'kmeans', 'leiden'):
            raise ValueError("method must be 'hdbscan', 'kmeans' or 'leiden'.")

# ClusteringResult: class containing labels, PCA embedding and centroids from a clustering
@dataclass
class ClusteringResult:
    labels: np.ndarray  # -1 means noise (HDBSCAN only)
    embedding: np.ndarray   # PCA coordinates
    explained_variance_ratio: np.ndarray
    centroids: dict[int, np.ndarray]    # label -> centroid in PCA space

# _leiden_labels: Leiden modularity communities on a symmetric kNN graph of the embedding, communities below min_cluster_size are noise (-1)
def _leiden_labels(embedding: np.ndarray, config: ClusteringConfig) -> np.ndarray:
    adjacency = kneighbors_graph(embedding, min(config.n_neighbors, embedding.shape[0] - 1), mode='connectivity')
    symmetric = adjacency.maximum(adjacency.T).tocoo()
    graph = igraph.Graph(n=embedding.shape[0], edges=list(zip(symmetric.row.tolist(), symmetric.col.tolist())))
    membership = np.array(leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=config.resolution, seed=config.random_state).membership)
    membership[np.bincount(membership)[membership] < config.min_cluster_size] = -1
    labels = np.full_like(membership, -1)
    for new, old in enumerate(np.unique(membership[membership >= 0])):   # renumber so labels run 0..K-1
        labels[membership == old] = new
    return labels

# _cluster_embedding: run the configured clusterer on an embedding, return (labels, centroids)
def _cluster_embedding(embedding: np.ndarray, config: ClusteringConfig) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    if config.method == 'hdbscan':
        if embedding.shape[0] < config.min_cluster_size:
            raise ValueError(f'only {embedding.shape[0]} particles but min_cluster_size is {config.min_cluster_size}')
        from fast_hdbscan import HDBSCAN
        labels = HDBSCAN(min_cluster_size=config.min_cluster_size, min_samples=max(config.min_cluster_size - 1, 1)).fit_predict(np.asarray(embedding, dtype=np.float64))
    elif config.method == 'leiden':
        labels = _leiden_labels(embedding, config)
    else:
        n_clusters = min(config.n_clusters, embedding.shape[0])
        labels = KMeans(n_clusters=n_clusters, random_state=config.random_state, n_init=10).fit_predict(embedding)
    centroids = {int(label): embedding[labels == label].mean(axis=0) for label in np.unique(labels) if label != -1}
    return labels, centroids

# reduce_features: PCA of a feature matrix, return (embedding, explained variance ratio)
def reduce_features(features: np.ndarray, config: ClusteringConfig) -> tuple[np.ndarray, np.ndarray]:
    if features.shape[0] == 0:
        raise ValueError('cannot cluster an empty feature matrix')
    n_components = min(config.n_components, features.shape[0] - 1, features.shape[1])
    if n_components < 2:
        raise ValueError(f'Too few particles ({features.shape[0]}) to run PCA (at least 2 needed)')
    log.progress(f'Starting PCA: {features.shape[0]} particles, {features.shape[1]} features, {n_components} components')
    pca = PCA(n_components=n_components, whiten=config.method == 'leiden', random_state=config.random_state)    # equal-variance axes keep one feature block from deciding the kNN graph
    embedding = pca.fit_transform(features)
    log.progress(f'PCA complete: {pca.explained_variance_ratio_.sum():.1%} variance explained')
    return embedding, pca.explained_variance_ratio_


# reduce_and_cluster: PCA followed by HDBSCAN, KMeans or Leiden
def reduce_and_cluster(features: np.ndarray, config: ClusteringConfig) -> ClusteringResult:
    embedding, explained_variance_ratio = reduce_features(features, config)
    log.progress(f'Starting {config.method} clustering')
    log_config(config.method, config)
    labels, centroids = _cluster_embedding(embedding, config)
    log.progress(f'Clustering complete: {len(set(labels) - {-1})} cluster(s), {int((labels == -1).sum())} noise')
    return ClusteringResult(labels=labels, embedding=embedding, explained_variance_ratio=explained_variance_ratio, centroids=centroids)

# label_to_cluster_id: turn a raw cluster label into a stable string id
def label_to_cluster_id(label: int) -> str:
    return NOISE_CLUSTER_ID if label == -1 else f'c{label:02d}'
