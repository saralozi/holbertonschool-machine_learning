#!/usr/bin/env python3
"""A function that builds a neural network with the Keras library"""

import tensorflow.keras as K
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """A function that builds a neural network with the Keras library"""

    model = K.models.Sequential()
    reg = K.regularizers.l2(lambtha)

    model.add(
        K.layers.Dense(
            units=layers[0],
            activation=activations[0],
            kernel_regularizer=reg,
            input_shape=(nx,)
        )
    )
    model.add(K.layers.Dropout(rate=1 - keep_prob))

    for i in range(1, len(layers)):
        model.add(
            K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=reg
            )
        )
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
