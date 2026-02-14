#!/usr/bin/env python3
"""A function that calculates the cost
 of a neural network with L2 regularization"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the total cost with L2 regularization"""
    layer_costs = []

    for layer in model.layers:
        if len(layer.trainable_weights) == 0:
            continue

        reg = tf.add_n(layer.losses) if layer.losses else 0.0
        layer_costs.append(cost + reg)

    return tf.stack(layer_costs)
