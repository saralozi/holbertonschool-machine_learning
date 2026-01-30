#!/usr/bin/env python3
"""A function that builds a neural network with the Keras library"""

import tensorflow as tf
from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """A function that builds a neural network with the Keras library"""

    if len(layers) != len(activations):
        raise ValueError("layers and activations must have the same length")

    inputs = keras.Input(shape=(nx,))

    x = inputs

    l2_reg = keras.regularizers.l2(lambtha)

    for i in range(len(layers)):

        x = keras.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=l2_reg
        )(x)

        if i != len(layers) - 1:
            x = keras.layers.Dropout(rate=1 - keep_prob)(x)

    model = keras.Model(inputs=inputs, outputs=x)

    return model
