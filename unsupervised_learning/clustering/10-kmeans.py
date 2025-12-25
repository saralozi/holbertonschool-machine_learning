#!/usr/bin/env python3
"""A function def kmeans(X, k): that performs K-means on a dataset"""


import sklearn.cluster


def kmeans(X, k):
    """A that performs K-means on a dataset"""

    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)

    return kmeans.cluster_centers_, kmeans.labels_
