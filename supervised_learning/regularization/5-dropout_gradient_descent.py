#!/usr/bin/env python3
"""A function that updates the weights of a neural network with Dropout regularization using gradient descent"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Gradient descent with dropout regularization"""
    m = Y.shape[1]

    dZ = cache["A{}".format(L)] - Y

    for l in range(L, 0, -1):
        A_prev = cache["A{}".format(l - 1)]
        W = weights["W{}".format(l)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights["W{}".format(l)] = W - alpha * dW
        weights["b{}".format(l)] = weights["b{}".format(l)] - alpha * db

        if l > 1:
            dA_prev = np.matmul(W.T, dZ)

            D_prev = cache["D{}".format(l - 1)]
            dA_prev = (dA_prev * D_prev) / keep_prob

            A_prev_layer = cache["A{}".format(l - 1)]
            dZ = dA_prev * (1 - np.square(A_prev_layer))
