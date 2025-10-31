#!/usr/bin/env python3
"""A script that takes a DataFrame as input and performs operations"""


import pandas as pd
import numpy as np


def array(df):
    """A function that takes a DataFrame and performs operations"""

    selected_rows = df[['High', 'Close']].tail(10)
    rows_array = np.array(selected_rows)
    return rows_array
