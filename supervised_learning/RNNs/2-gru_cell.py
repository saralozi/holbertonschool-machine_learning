#!/usr/bin/env python3
"""GRU Cell"""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """Class constructor"""

        # Update gate
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        # Reset gate
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        # Candidate hidden state
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        # Output layer
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation"""

        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Forward propagation for one time step"""

        # Concatenate previous hidden state and current input
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)

        # Reset gate
        r = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        # Reset applied to previous hidden state
        reset_h = r * h_prev

        # Concatenate reset hidden state with input
        concat_reset = np.concatenate((reset_h, x_t), axis=1)

        # Candidate hidden state
        h_hat = np.tanh(np.matmul(concat_reset, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_hat

        # Output
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
