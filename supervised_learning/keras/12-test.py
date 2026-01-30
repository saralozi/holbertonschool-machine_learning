#!/usr/bin/env python3
"""A function that tests a neural network model"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network model"""

    v = 1 if verbose else 0
    loss, acc = network.evaluate(data, labels, verbose=v)
    return loss, acc
