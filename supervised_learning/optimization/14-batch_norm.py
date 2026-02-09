#!/usr/bin/env python3
"""Creates a batch normalization layer in TensorFlow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a dense layer followed by batch norm and activation"""

    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    Z = tf.keras.layers.Dense(units=n, kernel_initializer=init)(prev)

    mean, var = tf.nn.moments(Z, axes=[0], keepdims=True)
    gamma = tf.Variable(tf.ones((1, n)), trainable=True)
    beta = tf.Variable(tf.zeros((1, n)), trainable=True)

    Z_norm = (Z - mean) / tf.sqrt(var + 1e-7)
    Z_tilde = gamma * Z_norm + beta

    return activation(Z_tilde)
