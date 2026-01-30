#!/usr/bin/env python3
"""A class that defines a deep neural
 network performing binary classification"""

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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx
        for layer, nodes in enumerate(layers, start=1):
            self.__weights["W{}".format(layer)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )
            self.__weights["b{}".format(layer)] = np.zeros((nodes, 1))
            prev = nodes

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation"""

        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]
            A_prev = self.__cache["A{}".format(layer - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(layer)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost using logistic regression"""

        m = Y.shape[1]
        cost = -np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        ) / m

        return cost

    def evaluate(self, X, Y):
        """Evaluates the network's predictions"""

        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = (A >= 0.5).astype(int)

        return prediction, cost
