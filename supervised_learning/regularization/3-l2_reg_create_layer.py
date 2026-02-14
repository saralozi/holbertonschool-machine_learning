#!/usr/bin/env python3
"""A function that creates a neural
network layer in tensorflow with L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a layer with L2 regularization """

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.HeNormal(),
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )

    return layer(prev)
