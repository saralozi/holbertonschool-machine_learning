#!/usr/bin/env python3
"""Deep neural network class (multiclass) with persistence"""

import os
import pickle
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for multiclass classification"""

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
        """Calculates forward propagation (sigmoid hidden, softmax output)"""

        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]
            A_prev = self.__cache["A{}".format(layer - 1)]

            Z = np.matmul(W, A_prev) + b

            if layer == self.__L:
                # Softmax (stable)
                Z_shift = Z - np.max(Z, axis=0, keepdims=True)
                exp_Z = np.exp(Z_shift)
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # Sigmoid for hidden layers
                A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(layer)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates cost using categorical cross-entropy"""

        m = Y.shape[1]
        A = A.astype(np.float64)
        A = np.clip(A, 1e-300, 1.0)
        return -np.sum(Y * np.log(A)) / m


    def evaluate(self, X, Y):
        """Evaluates predictions: returns one-hot predictions and cost"""

        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        classes = A.shape[0]
        m = A.shape[1]

        prediction = np.zeros((classes, m))
        pred_idx = np.argmax(A, axis=0)
        prediction[pred_idx, np.arange(m)] = 1

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the network"""

        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y

        for layer in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(layer - 1)]
            W_key = "W{}".format(layer)
            b_key = "b{}".format(layer)
            W = self.__weights[W_key]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if layer > 1:
                dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))

            self.__weights[W_key] = self.__weights[W_key] - alpha * dW
            self.__weights[b_key] = self.__weights[b_key] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        # 0th iteration (before training)
        A, cache = self.forward_prop(X)
        c0 = self.cost(Y, A)

        if verbose:
            print("Cost after 0 iterations: {}".format(c0))
        if graph:
            steps.append(0)
            costs.append(c0)

        for i in range(1, iterations + 1):
            self.gradient_descent(Y, cache, alpha)
            A, cache = self.forward_prop(X)

            if (verbose or graph) and (i % step == 0 or i == iterations):
                c = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
                if graph:
                    steps.append(i)
                    costs.append(c)

        if graph:
            plt.plot(steps, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""

        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""

        if not os.path.exists(filename):
            return None

        with open(filename, "rb") as f:
            return pickle.load(f)
