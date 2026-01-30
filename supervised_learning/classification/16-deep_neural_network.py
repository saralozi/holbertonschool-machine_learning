#!/usr/bin/env python3
"""A class that defines a deep neural
network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """
        nx: number of input features
        layers: list of nodes in each layer
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for nodes in layers:
            if type(nodes) is not int or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        prev = nx
        for layer, nodes in enumerate(layers, start=1):
            self.weights[f"W{layer}"] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )
            self.weights[f"b{layer}"] = np.zeros((nodes, 1))
            prev = nodes
