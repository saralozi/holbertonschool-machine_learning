#!/usr/bin/env python3
"""A function that creates a layer of a neural network using dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a layer with dropout"""

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.HeNormal()
    )(prev)

    drop = tf.keras.layers.Dropout(rate=1 - keep_prob)(dense, training=training)

    return drop
