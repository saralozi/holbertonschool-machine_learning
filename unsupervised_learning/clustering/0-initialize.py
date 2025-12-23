#!/usr/bin/env python3
"""A function def initialize(X, k): that initializes
cluster centroids for K-means"""


import numpy as np


def initialize(X, k):
    """"A function that initializes cluster centroids for K-Means"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    return np.random.uniform(low=min_vals, high=max_vals, size=(k, X.shape[1]))
