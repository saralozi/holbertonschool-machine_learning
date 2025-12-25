#!/usr/bin/env python3
"""A function def pca(X, ndim): that performs PCA on a dataset"""


import numpy as np


def pca(X, ndim):
    """A function that performs PCA on a dataset"""

    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(ndim, int) or ndim <= 0 or ndim > X.shape[1]):
        return None

    X_centered = X - np.mean(X, axis=0)

    cov = np.cov(X_centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx[:ndim]]

    T = X_centered @ W
    return T
