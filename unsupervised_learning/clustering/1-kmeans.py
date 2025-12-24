#!/usr/bin/env python3
"""A function def kmeans(X, k, iterations=1000)
that performs K-means on a dataset"""


import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means"""

    _, d = X.shape

    centroids = np.random.uniform(low=np.min(
        X, axis=0), high=np.max(X, axis=0), size=(k, d))

    return centroids


def kmeans(X, k, iterations=1000):
    """A function that performs K-Means on a dataset"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = initialize(X, k)

    for _ in range(iterations):
        C_prev = np.copy(C)
        sum_cluster_points = np.zeros_like(C)
        n_cluster_points = np.zeros((k, 1))

        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=-1)
        clss = np.argmin(distances, axis=1)

        for i in range(k):
            cluster_points = X[clss == i]
            if len(cluster_points) == 0:
                C[i] = initialize(X, 1)[0]
            else:
                sum_cluster_points[i] = np.sum(cluster_points, axis=0)
                n_cluster_points[i] = cluster_points.shape[0]

        non_empty_clusters = n_cluster_points.flatten() != 0
        C[non_empty_clusters] = sum_cluster_points[non_empty_clusters] / \
            n_cluster_points[non_empty_clusters]

        if np.array_equal(C, C_prev):
            break

    return C, clss
