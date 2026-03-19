#!/usr/bin/env python3
"""Write a function that builds a dense block"""


from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block"""

    initializer = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        # Bottleneck layer
        Y = K.layers.BatchNormalization(axis=3)(X)
        Y = K.layers.Activation('relu')(Y)
        Y = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            kernel_initializer=initializer
        )(Y)

        Y = K.layers.BatchNormalization(axis=3)(Y)
        Y = K.layers.Activation('relu')(Y)
        Y = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_initializer=initializer
        )(Y)

        # Concatenate new features with previous features
        X = K.layers.Concatenate(axis=3)([X, Y])

        # Update number of filters
        nb_filters += growth_rate

    return X, nb_filters
