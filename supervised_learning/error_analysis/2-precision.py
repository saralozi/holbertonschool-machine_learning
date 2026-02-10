#!/usr/bin/env python3
"""A function that calculates
 the precision for each class in a confusion matrix"""

import numpy as np


def precision(confusion):
    """Precision for each class in confusion matrix"""

    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    precision = true_positives / predicted_positives

    return precision
