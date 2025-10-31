#!/usr/bin/env python3
"""A script that takes a DataFrame and does slicing operations on it"""


def slice(df):
    """A function that performs slicing operations"""

    cols = ['High', 'Low', 'Close']

    if 'Volume_BTC' in df.columns:
        cols.append('Volume_BTC')
    elif 'Volume_(BTC)' in df.columns:
        cols.append('Volume_(BTC)')
    else:
        raise KeyError(
            "Neither 'Volume_BTC' nor 'Volume_(BTC)' found in columns"
        )

    return df[cols].iloc[::60, :]
