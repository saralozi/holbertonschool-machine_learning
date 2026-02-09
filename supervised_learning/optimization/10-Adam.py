#!/usr/bin/env python3
"""A function that sets up the Adam optimization algorithm in TensorFlow"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Sets up Adam optimization in TensorFlow"""

    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
