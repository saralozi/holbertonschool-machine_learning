#!/usr/bin/env python3
"""A function def expectation(X, pi, m, S):
that calculates the expectation step in the EM algorithm for a GMM:"""

import numpy as np
pdf = __import__("5-pdf").pdf


def expectation(X, pi, m, S):
    """A function that calculates expectation step in EM algorithm for GMM"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    n, d = X.shape

    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    k = pi.shape[0]

    if not isinstance(m, np.ndarray) or m.shape != (k, d):
        return None, None

    if not isinstance(S, np.ndarray) or S.shape != (k, d, d):
        return None, None

    if np.any(pi < 0) or not np.isclose(np.sum(pi), 1):
        return None, None

    P = np.array([pdf(X, m[i], S[i]) for i in range(k)])
    if P is None:
        return None, None

    weighted = pi[:, None] * P
    total = np.sum(weighted, axis=0)

    g = weighted / total
    log_likelihood = np.sum(np.log(total))

    return g, log_likelihood
