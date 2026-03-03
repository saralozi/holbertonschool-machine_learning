#!/usr/bin/env python3
"""Back propagation over a convolutional layer"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer"""

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape

    if padding == "valid":
        ph, pw = 0, 0
    elif padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    A_pad = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0
    )
    dA_pad = np.zeros_like(A_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            y = i * sh
            x = j * sw

            A_slice = A_pad[:, y:y + kh, x:x + kw, :]

            for k in range(c_new):
                dW[:, :, :, k] += np.sum(
                    A_slice * dZ[:, i:i + 1, j:j + 1, k:k + 1],
                    axis=0
                )

                grad = W[:, :, :, k] * dZ[:, i:i + 1, j:j + 1, k:k + 1]
                dA_pad[:, y:y + kh, x:x + kw, :] += grad

    if ph == 0 and pw == 0:
        dA_prev = dA_pad
    else:
        dA_prev = dA_pad[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
