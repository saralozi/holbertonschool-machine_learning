#!/usr/bin/env python3
"""A script that takes a DataFrame and computes statistics"""


def analyze(df):
    """A function that takes a DataFrame and computes statistics"""

    new_df = df.drop('Timestamp', axis=1)
    statistics = new_df.describe()
    return statistics
