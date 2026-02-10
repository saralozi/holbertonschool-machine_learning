#!/usr/bin/env python3
"""A function that creates a confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Build confusion matrix"""

    return np.matmul(labels.T, logits)
