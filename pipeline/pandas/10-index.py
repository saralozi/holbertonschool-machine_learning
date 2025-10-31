#!/usr/bin/env python3
"""A script that takes a DataFrame and changes index"""


def index(df):
    """A function that takes a DataFrame and changes index"""

    new_df = df.set_index('Timestamp')
    return new_df
