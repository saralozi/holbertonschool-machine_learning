#!/usr/bin/env python3
"""A function that calculates the
cost of a neural network with L2 regularization"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculate cost of neural network with L2 regularization"""

    l2_sum = 0

    for i in range(1, L + 1):
        W = weights["W{}".format(i)]
        l2_sum += np.sum(np.square(W))

    l2_term = (lambtha / (2 * m)) * l2_sum

    return cost + l2_term
