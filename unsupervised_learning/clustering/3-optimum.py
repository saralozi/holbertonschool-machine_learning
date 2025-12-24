#!/usr/bin/env python3
""" A function that tests for the optimum number of clusters by variance:"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """A function that tests for the optimum of clusters by variance"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if kmax <= kmin:
        return None, None

    results = []
    d_vars = []

    C0, clss0 = kmeans(X, kmin, iterations)
    if C0 is None:
        return None, None
    var0 = variance(X, C0)

    for k in range(kmin, kmax + 1):
        C, clss = (C0, clss0) if k == kmin else kmeans(X, k, iterations)
        if C is None:
            return None, None
        results.append((C, clss))
        d_vars.append(var0 - variance(X, C))

    return results, d_vars
