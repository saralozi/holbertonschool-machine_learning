#!/usr/bin/env python3
"""A script that takes a DataFrame and does delete operations"""


def prune(df):
    """A function that does delete operations"""

    new_data = df.dropna(subset=['Close'])
    return new_data
