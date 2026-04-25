#!/usr/bin/env python3
"""LSTM Cell"""

import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """Class constructor"""

        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))

        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))

        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))

        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))

        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def softmax(self, Y):
        """Calculates softmax"""

        exp = np.exp(Y - np.max(Y, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def sigmoid(self, x):
        """Calculates sigmoid"""

        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """Forward propagation for one time step"""

        concat = np.concatenate((h_prev, x_t), axis=1)

        f = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        u = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        c_bar = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_bar

        o = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)
        h_next = o * np.tanh(c_next)

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
