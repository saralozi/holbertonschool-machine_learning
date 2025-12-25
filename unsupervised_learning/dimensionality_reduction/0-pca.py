#!/usr/bin/env python3
"""A function def pca(X, var=0.95): that performs PCA on a dataset"""


import numpy as np


def pca(X, var=0.95):
    """A function that performs PCA on a dataset"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(var, float) or var <= 0 or var > 1:
        return None

    n, d = X.shape

    cov = (X.T @ X) / (n - 1)

    eig_vals, eig_vecs = np.linalg.eigh(cov)

    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    cum = np.cumsum(eig_vals)
    total = cum[-1]
    nd = np.searchsorted(cum / total, var) + 1

    W = eig_vecs[:, :nd]

    return W
