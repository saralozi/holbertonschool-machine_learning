#!/usr/bin/env python3
"""A function that shuffles the data points in two matrices the same way"""

import numpy as np


def shuffle_data(X, Y):
    """Shuffles X and Y using same permutation"""

    permutation = np.random.permutation(X.shape[0])

    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    return X_shuffled, Y_shuffled
