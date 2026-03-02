#!/usr/bin/env python3
"""A function that performs forward propagation
over a convolutional layer of a neural network.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """Perform forward propagation over a convolutional layer"""

    # Extract shapes
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Decide padding amount
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2))

    # Compute output height and width
    output_height = int((h_prev - kh + 2 * ph) / sh + 1)
    output_width = int((w_prev - kw + 2 * pw) / sw + 1)

    # Create an output container
    # Store the convolution result before activation
    convolved_images = np.zeros((m, output_height, output_width, c_new))

    # Pad the input images
    image_pad = np.pad(A_prev,
                       ((0, 0), (ph, ph),
                        (pw, pw), (0, 0)), mode='constant')

    # Compute every output pixel for every output channel
    for k in range(c_new):
        for h in range(output_height):
            for w in range(output_width):
                # Grab the sliding “window” from every image at once
                image_zone = image_pad[:, h * sh:h * sh + kh,
                                       w * sw:w * sw + kw, :]

                # Multiply by filter and sum to get a number
                convolved_images[:, h, w, k] = np.sum(image_zone
                                                      * W[:, :, :, k],
                                                      axis=(1, 2, 3))
    # Add bias
    Z = convolved_images + b
    # Apply activation
    Z = activation(Z)

    return Z
