#!/usr/bin/env python3
"""A function that creates a batch normalization
 layer for a neural network in tensorflow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a dense layer followed by batch normalization"""

    dense = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'
        )
    )(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer=tf.ones_initializer(),
        beta_initializer=tf.zeros_initializer()
    )(dense)

    output = activation(batch_norm)

    return output
