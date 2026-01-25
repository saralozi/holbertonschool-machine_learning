#!/usr/bin/env python3
"""A class Neuron that defines a single neuron
performing binary classification with private instance attributes"""


import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron"""

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
