#!/usr/bin/env python3
"""Write a function that builds an inception block"""


from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Write a function that builds an inception block"""

    F1, F3R, F3, F5R, F5, FPP = filters

    branch1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    branch2 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    branch2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(branch2)

    branch3 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    branch3 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )(branch3)

    branch4 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    branch4 = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(branch4)

    output = K.layers.Concatenate(axis=-1)(
        [branch1, branch2, branch3, branch4]
    )

    return output
