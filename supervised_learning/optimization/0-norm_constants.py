#!/usr/bin/env python3
"""A function that calculates the normalization constants of a matrix"""


import numpy as np


def normalization_constants(X):
    """Calculates mean and standard deviation per feature"""

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
