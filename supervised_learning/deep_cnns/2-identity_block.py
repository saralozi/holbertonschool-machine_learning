"""Write a function that builds an identity block"""


from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Write a function that builds an identity block"""

    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    # First component of main path
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)

    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)

    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)

    X = K.layers.BatchNormalization(axis=3)(X)

    # Add shortcut value to main path
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
