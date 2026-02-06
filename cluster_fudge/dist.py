import numpy as np
import numpy.typing as npt
from numba import prange, njit
from enum import Enum


class DistanceMetrics(Enum):
    HAMMING = "hamming"
    JACCARD = "jaccard"
    NG = "ng"


# we have list of centroid which we want to compare our input to
# a point is a list of xyz, in higher dimensions it has n items
# we have a list of centroids we want to compare against a list of targets


# X is data, Centroids is centroids
@njit(parallel=True, fastmath=True)
def hamming(X: np.ndarray, centroids: np.ndarray) -> npt.NDArray[np.float64]:
    """
    Compute Hamming distance between X and centroids.

    Args:
        X: (npt.NDArray[np.int64]) Data array (n_samples, n_features)
        centroids: (npt.NDArray[np.int64]) Centroids array (n_clusters, n_features)

    Returns:
        (npt.NDArray[np.float64]) Distance matrix (n_samples, n_clusters)
    """

    if X.shape[1] != centroids.shape[1]:
        raise ValueError("X and centroids must have the same number of features")

    rows = X.shape[0]  # NUMBER of rows
    n_clusters = centroids.shape[
        0
    ]  # number of columns of distance matrix = number of centroids
    distance: npt.NDArray[np.float64] = np.zeros(
        (rows, n_clusters), dtype=int
    )  # matrix
    for i in prange(rows):  # iterate from 0 through the number of rows
        for j in range(n_clusters):
            dist = 0
            for a, b in zip(X[i], centroids[j]):
                if a != b:
                    dist += 1
            distance[i][j] = dist
    return distance


@njit(parallel=True, fastmath=True)
def jaccard(X: np.ndarray, centroids: np.ndarray) -> npt.NDArray[np.float64]:
    """
    Compute Jaccard distance between X and centroids.

    Args:
        X: (npt.NDArray[np.int64]) Data array (n_samples, n_features)
        centroids: (npt.NDArray[np.int64]) Centroids array (n_clusters, n_features)

    Returns:
        (npt.NDArray[np.float64]) Distance matrix (n_samples, n_clusters)
    """

    if X.shape[1] != centroids.shape[1]:
        raise ValueError("X and centroids must have the same number of features")

    rows, cols = X.shape  # NUMBER of rows
    n_clusters = centroids.shape[
        0
    ]  # number of columns of distance matrix = number of centroids
    distance = np.zeros((rows, n_clusters), dtype=int)  # matrix
    for i in prange(rows):  # iterate from 0 through the number of rows
        for j in range(n_clusters):
            dist = 0
            for a, b in zip(X[i], centroids[j]):
                if a == b:
                    dist += 1
            distance[i][j] = (
                1 - (dist / (cols * 2 - dist))
            )  # double the num of columns (because a union between both centroids and data points), then subtract the numner of similar elements
    return distance


def ng(X, centroids):
    pass


def distance(
    X: np.ndarray, centroids: np.ndarray, metric: DistanceMetrics
) -> npt.NDArray[np.float64]:
    """
    Compute distance between X and centroids using the specified metric.

    Args:
        X: (npt.NDArray[np.int64]) Data array (n_samples, n_features)
        centroids: (npt.NDArray[np.int64]) Centroids array (n_clusters, n_features)
        metric: (DistanceMetrics) Distance metric to use

    Returns:
        (npt.NDArray[np.float64]) Distance matrix (n_samples, n_clusters)
    """

    if metric == DistanceMetrics.HAMMING:
        return hamming(X, centroids)
    elif metric == DistanceMetrics.JACCARD:
        return jaccard(X, centroids)
    elif metric == DistanceMetrics.NG:
        return ng(X, centroids)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
