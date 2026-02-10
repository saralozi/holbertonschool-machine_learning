#!/usr/bin/env python3
"""A function that calculates
 the specificity for each class in a confusion matrix"""

import numpy as np


def specificity(confusion):
    """Calculates specificity for each class"""

    cm = confusion.astype(float)

    total = np.sum(cm)

    fn = np.sum(cm, axis=1)
    fp = np.sum(cm, axis=0)

    tp = np.diag(cm)
    tn = total - fn - fp + tp

    specificity_values = tn / (tn + fp - tp)

    return specificity_values
