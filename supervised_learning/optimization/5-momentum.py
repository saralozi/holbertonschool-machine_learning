#!/usr/bin/env python3
"""A function that updates a variable using
 the gradient descent with momentum optimization algorithm"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Performs momentum update"""

    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v

    return var, v
