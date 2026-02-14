#!/usr/bin/env python3
"""A function that creates a layer of a neural network using dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a layer with dropout"""

    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    dense_layer = tf.keras.layers.Dense(units=n, activation=activation,
                                        kernel_initializer=init)

    output = dense_layer(prev)

    dropout_layer = tf.keras.layers.Dropout(rate=1-keep_prob)
    output = dropout_layer(output, training=training)

    return output
