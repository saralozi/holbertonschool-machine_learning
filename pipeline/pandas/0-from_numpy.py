#!/usr/bin/env python3
"""A script that creates a DataFrame from an array"""


import pandas as pd


def from_numpy(array):
    """A function that creates a Pandas DataFrame from a NumPy array"""

    cols = [chr(65 + i) for i in range(array.shape[1])]
    df = pd.DataFrame(array, columns=cols)
    return df
