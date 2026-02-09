#!/usr/bin/env python3
"""A function that normalizes (standardizes) a matrix"""

import numpy as np


def normalize(X, m, s):
    """Standardizes matrix X"""

    X_norm = (X - m) / s
    return X_norm
