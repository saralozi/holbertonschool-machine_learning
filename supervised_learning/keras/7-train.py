#!/usr/bin/env python3
"""A function that trains a model using mini-batch gradient
descent with optional validation, early stopping, and learning rate decay"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent with
    optional validation, early stopping, and learning rate decay"""

    callbacks = []

    has_val = validation_data is not None

    if early_stopping and has_val:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
        )

    if learning_rate_decay and has_val:
        def schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(
            K.callbacks.LearningRateScheduler(
                schedule,
                verbose=1
            )
        )

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks if callbacks else None
    )

    return history
