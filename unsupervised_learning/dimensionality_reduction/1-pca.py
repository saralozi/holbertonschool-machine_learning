#!/usr/bin/env python3
"""A function def pca(X, ndim): that performs PCA on a dataset"""


import numpy as np


def pca(X, ndim):
    """A function that performs PCA on a dataset"""

    X_m = X - np.mean(X, axis=0)
    U, S, Vh = np.linalg.svd(X_m)
    W = Vh.T[:, :ndim]

    return X_m @ W
