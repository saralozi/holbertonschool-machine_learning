#!/usr/bin/env python3
"""A function that calculates the cost of
 a neural network with L2 regularization"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the cost with L2 regularization"""

    reg_losses = model.losses

    return cost + tf.stack(reg_losses)
