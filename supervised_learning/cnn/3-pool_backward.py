#!/usr/bin/env python3
"""Back propagation over a pooling layer"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Back propagation over a pooling layer"""

    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    _, h_new, w_new, _ = dA.shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            y = i * sh
            x = j * sw

            window = A_prev[:, y:y + kh, x:x + kw, :]

            if mode == 'avg':
                da = dA[:, i:i + 1, j:j + 1, :]
                dA_prev[:, y:y + kh, x:x + kw, :] += da / (kh * kw)

            elif mode == 'max':
                max_vals = np.max(window, axis=(1, 2), keepdims=True)
                mask = (window == max_vals).astype(float)

                da = dA[:, i:i + 1, j:j + 1, :]
                dA_prev[:, y:y + kh, x:x + kw, :] += mask * da

            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return dA_prev
