#!/usr/bin/env python3
"""A function that calculates
 the sensitivity for each class in a confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """Calculates sensitivity (recall) for each class"""

    cm = confusion.astype(float)

    true_positives = np.diag(cm)
    false_negatives = cm.sum(axis=1) - true_positives

    sensitivity_values = true_positives / (true_positives + false_negatives)

    return sensitivity_values
