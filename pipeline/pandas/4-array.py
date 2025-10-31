#!/usr/bin/env python3
"""A script that takes a DataFrame as input and performs operations"""


def array(df):
    """A function that takes a DataFrame and performs operations"""

    selected_rows = df[['High', 'Close']].tail(10)
    rows_array = selected_rows.to_numpy()
    return rows_array
