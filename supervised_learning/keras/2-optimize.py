#!/usr/bin/env python3
"""Sets up Adam optimization for a keras model."""

import tensorflow.keras as K

def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for a keras model."""

    opt = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    if not hasattr(opt, "lr") and hasattr(opt, "learning_rate"):
        opt.lr = opt.learning_rate

    network.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
