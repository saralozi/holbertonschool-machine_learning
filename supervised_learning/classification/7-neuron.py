#!/usr/bin/env python3
"""A class Neuron that defines a single neuron
performing binary classification with private instance attributes"""

import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""

        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost using logistic regression."""

        m = Y.shape[1]
        cost = -np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        ) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions"""

        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate one pass of gradient descent on the neuron"""

        m = Y.shape[1]
        dz = A - Y
        dW = np.matmul(dz, X.T) / m
        db = np.sum(dz) / m

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """Train the neuron by updating private attributes"""

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        xs = []
        ys = []

        A0 = self.forward_prop(X)
        c0 = self.cost(Y, A0)
        if verbose:
            print("Cost after 0 iterations: {}".format(c0))
        if graph:
            xs.append(0)
            ys.append(c0)

        for i in range(1, iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if (verbose or graph) and (i % step == 0 or i == iterations):
                Ai = self.forward_prop(X)
                ci = self.cost(Y, Ai)

                if verbose:
                    print("Cost after {} iterations: {}".format(i, ci))
                if graph:
                    xs.append(i)
                    ys.append(ci)

        if graph:
            plt.plot(xs, ys, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
