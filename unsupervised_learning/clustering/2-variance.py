#!/usr/bin/env python3
"""Write a function def variance(X, C): that calculates
the total intra-cluster variance for a data set"""

import numpy as np


def variance(X, C):
    """A function that calculates the
    total intra-cluster variance for a dataset"""

    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or not isinstance(C, np.ndarray)
        or C.ndim != 2
        or C.shape[1] != X.shape[1]
    ):
        return None

    distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=-1)
    clss = np.argmin(distances, axis=1)

    cluster_distances = distances[np.arange(len(X)), clss]
    var = np.sum(cluster_distances ** 2)

    return var
