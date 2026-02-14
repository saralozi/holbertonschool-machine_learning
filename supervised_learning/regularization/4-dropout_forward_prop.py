#!/usr/bin/env python3
"""A function that conducts forward propagation using Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Forward propagation with dropout"""

    cache = {'A0': X}

    for layer in range(1, L):
        Z = (np.matmul(weights["W" + str(layer)],
                       cache['A' + str(layer - 1)]) +
             weights['b' + str(layer)])
        A = np.tanh(Z)
        dropout = np.random.binomial(1, keep_prob, size=A.shape)
        cache["D" + str(layer)] = dropout
        A = np.multiply(A, dropout)
        A /= keep_prob
        cache['A' + str(layer)] = A

    Z = (np.matmul(weights["W" + str(L)],
                   cache['A' + str(L - 1)]) +
         weights['b' + str(L)])

    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)

    cache['A' + str(L)] = A

    return cache
