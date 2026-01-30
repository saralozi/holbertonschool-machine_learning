#!/usr/bin/env python3
"""A function that saves entire model"""

import tensorflow.keras as K


def save_model(network, filename):
    """Save the model"""

    network.save(filename)


def load_model(filename):
    """Load the model"""

    return K.models.load_model(filename)
