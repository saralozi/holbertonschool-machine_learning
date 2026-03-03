#!/usr/bin/env python3
"Performs a convolution on grayscale images"


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    "Performs a convolution on grayscale images"

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

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

    if padding == 'same':
        padded = np.pad(
            images,
            pad_width=((0, 0), (ph, ph + ph_extra), (pw, pw + pw_extra)),
            mode='constant',
            constant_values=0
        )
    else:
        padded = np.pad(
            images,
            pad_width=((0, 0), (ph, ph), (pw, pw)),
            mode='constant',
            constant_values=0
        )

    h_pad, w_pad = padded.shape[1], padded.shape[2]
    out_h = ((h_pad - kh) // sh) + 1
    out_w = ((w_pad - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            y = i * sh
            x = j * sw
            patch = padded[:, y:y + kh, x:x + kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
