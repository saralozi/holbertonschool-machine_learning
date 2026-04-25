#!/usr/bin/env python3
"""Deep RNN forward propagation"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Forward propagation for a deep RNN"""

    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        x_t = X[step]

        for layer in range(l):
            cell = rnn_cells[layer]

            h_prev = H[step, layer]

            if layer == 0:
                x_input = x_t
            else:
                x_input = H[step + 1, layer - 1]

            h_next, y = cell.forward(h_prev, x_input)
            H[step + 1, layer] = h_next

        Y[step] = y

    return H, Y
