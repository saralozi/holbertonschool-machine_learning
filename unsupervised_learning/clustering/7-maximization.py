#!/usr/bin/env python3
"""A function def maximization(X, g): that calculates
the maximization step in the EM algorithm for a GMM"""


import numpy as np


def maximization(X, g):
    """A function that calculates the maximization
    step in the EM algorithm for a GMM"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    n, d = X.shape

    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    k, n2 = g.shape
    if n2 != n:
        return None, None, None

    if not np.allclose(np.sum(g, axis=0), np.ones(n)):
        return None, None, None

    Nk = np.sum(g, axis=1)
    if np.any(Nk == 0):
        return None, None, None

    pi = Nk / n

    m = (g @ X) / Nk[:, None]

    X_centered = X[None, :, :] - m[:, None, :]
    S = (g[:, :, None] * X_centered).transpose(0, 2, 1) @ X_centered
    S = S / Nk[:, None, None]

    return pi, m, S
