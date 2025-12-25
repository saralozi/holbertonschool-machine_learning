#!/usr/bin/env python3
"""A function def pca(X, var=0.95): that performs PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(var, float) or var <= 0 or var > 1:
        return None

    # X is already mean-centered (task guarantees mean = 0 per column)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Explained variance is proportional to singular values squared
    explained = s ** 2
    cum_ratio = np.cumsum(explained) / np.sum(explained)

    # smallest nd such that we keep at least `var`
    nd = np.where(cum_ratio >= var)[0][0] + 1

    W = Vt.T[:, :nd]
    return W
