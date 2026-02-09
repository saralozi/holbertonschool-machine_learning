#!/usr/bin/env python3
"""A function that normalizes an unactivated output
 of a neural network using batch normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes Z using batch normalization"""

    mu = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)
    Z_batch = gamma * Z_norm + beta

    return Z_batch
