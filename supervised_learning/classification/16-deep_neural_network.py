#!/usr/bin/env python3
"""A class that defines a deep
 neural network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """Initialize the deep neural network"""

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if not all(map(lambda x: type(x) is int and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        prev = nx
        for layer, nodes in enumerate(layers, start=1):
            self.weights["W{}".format(layer)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )
            self.weights["b{}".format(layer)] = np.zeros((nodes, 1))
            prev = nodes
