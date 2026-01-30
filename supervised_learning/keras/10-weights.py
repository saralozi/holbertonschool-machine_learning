#!/usr/bin/env python3
"""A function that saves a model's weights"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Saves a model's weights"""

    network.save_weights(filename)


def load_weights(network, filename):
    """Load a model's weights"""

    network.load_weights(filename)
