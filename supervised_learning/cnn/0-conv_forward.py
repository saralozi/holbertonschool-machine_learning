#!/usr/bin/env python3
"""A function that performs forward propagation
over a convolutional layer of a neural network.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """Perform forward propagation over a convolutional layer"""

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2))

    output_height = int((h_prev - kh + 2 * ph) / sh + 1)
    output_width = int((w_prev - kw + 2 * pw) / sw + 1)

    convolved_images = np.zeros((m, output_height, output_width, c_new))

    image_pad = np.pad(A_prev,
                       ((0, 0), (ph, ph),
                        (pw, pw), (0, 0)), mode='constant')

    for k in range(c_new):
        for h in range(output_height):
            for w in range(output_width):
                image_zone = image_pad[:, h * sh:h * sh + kh,
                                       w * sw:w * sw + kw, :]

                convolved_images[:, h, w, k] = np.sum(image_zone
                                                      * W[:, :, :, k],
                                                      axis=(1, 2, 3))

    Z = convolved_images + b

    Z = activation(Z)

    return Z
