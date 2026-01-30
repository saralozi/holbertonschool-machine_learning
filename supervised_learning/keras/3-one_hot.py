#!/usr/bin/env python3
"""A function that converts a label vector into a one-hot matrix"""

import numpy as np


def one_hot(labels, classes=None):
    """A function that converts a label vector into a one-hot matrix"""

    labels = np.array(labels)

    if classes is None:
        classes = np.max(labels) + 1

    one_hot = np.zeros((labels.shape[0], classes))

    one_hot[np.arange(labels.shape[0]), labels] = 1

    return one_hot
