import numpy.typing as npt
import numpy as np
from cluster_fudge.utils import DistanceMetrics


class ClusterFudge():
    def __init__(self, n_clusters: int = 8, n_init: int = 10, max_iter: int = 100, dist_metric:DistanceMetrics = DistanceMetrics.HAMMING) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.dist_metric = dist_metric
        self.centroids = None
        self.labels = None

    def fit(self, X:npt.NDArray[np.float64]) -> None:
        self.centroids = np.random.rand(self.n_clusters, X.shape[1])
        self.labels = np.zeros(X.shape[0], dtype=int)
