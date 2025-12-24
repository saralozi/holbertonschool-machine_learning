#!/usr/bin/env python3
"""A function def kmeans(X, k, iterations=1000)
that performs K-means on a dataset"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """A function that performs K-Means on a dataset"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    pool = np.random.uniform(
        low=min_vals, high=max_vals, size=(iterations * k, d))
    pool_i = 0

    for _ in range(iterations):
        C_prev = C.copy()

        dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        clss = dist.argmin(axis=1)

        counts = np.bincount(clss, minlength=k)
        sums = np.zeros((k, d))
        np.add.at(sums, clss, X)
        C = sums / np.where(counts[:, None] == 0, 1, counts[:, None])

        empty = counts == 0
        if np.any(empty):
            m = empty.sum()
            C[empty] = pool[pool_i:pool_i + m]
            pool_i += m

        if np.array_equal(C, C_prev):
            break

    return C, clss
