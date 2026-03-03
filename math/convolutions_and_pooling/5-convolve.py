#!/usr/bin/env python3
"""Performs a convolution on images using multiple kernels"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels"""

    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("kernels channels must match images channels")

    if padding == 'valid':
        ph, pw = 0, 0
        ph_extra, pw_extra = 0, 0

    elif padding == 'same':
        out_h = int(np.ceil(h / sh))
        out_w = int(np.ceil(w / sw))

        ph_total = max((out_h - 1) * sh + kh - h, 0)
        pw_total = max((out_w - 1) * sw + kw - w, 0)

        ph = ph_total // 2
        pw = pw_total // 2
        ph_extra = ph_total - 2 * ph
        pw_extra = pw_total - 2 * pw

    else:
        ph, pw = padding
        ph_extra, pw_extra = 0, 0

    padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph + ph_extra), (pw, pw + pw_extra), (0, 0)),
        mode='constant',
        constant_values=0
    )

    h_pad, w_pad = padded.shape[1], padded.shape[2]
    out_h = ((h_pad - kh) // sh) + 1
    out_w = ((w_pad - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            y = i * sh
            x = j * sw
            patch = padded[:, y:y + kh, x:x + kw, :]

            for k in range(nc):
                kern = kernels[:, :, :, k]
                output[:, i, j, k] = np.sum(patch * kern, axis=(1, 2, 3))

    return output
