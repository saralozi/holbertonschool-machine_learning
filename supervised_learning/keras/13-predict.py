#!/usr/bin/env python3
"""A function that makes predictions using a neural network"""


def predict(network, data, verbose=False):
    """Makes predictions using a neural network"""

    v = 1 if verbose else 0
    return network.predict(data, verbose=v)
