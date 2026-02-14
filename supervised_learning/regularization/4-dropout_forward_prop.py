#!/usr/bin/env python3
"""A function that conducts forward propagation using Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Forward propagation with dropout"""
    cache = {}
    cache["A0"] = X
    A = X

    for i in range(1, L + 1):
        W = weights["W{}".format(l)]
        b = weights["b{}".format(l)]
        Z = np.matmul(W, A) + b

        if i != L:
            A = np.tanh(Z)

            D = (np.random.rand(*A.shape) < keep_prob).astype(int)

            A = (A * D) / keep_prob

            cache["D{}".format(i)] = D
            cache["A{}".format(l)] = A
        else:
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = expZ / np.sum(expZ, axis=0, keepdims=True)
            cache["A{}".format(i)] = A

    return cache
