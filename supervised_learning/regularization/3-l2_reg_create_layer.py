#!/usr/bin/env python3
"""A function that calculates the
 cost of a neural network with L2 regularization"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculate cost of neural network with L2 regularzation"""
    layer_costs = []

    for layer in model.layers:
        if layer.losses:
            reg = tf.add_n(layer.losses)
            layer_costs.append(cost + reg)

    return tf.stack(layer_costs)
