#!/usr/bin/env python3
"""A function that sets up the gradient
 descent with momentum optimization algorithm in TensorFlow"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Sets up gradient descent with momentum in TensorFlow"""

    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
