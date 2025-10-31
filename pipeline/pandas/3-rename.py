#!usr/bin/env python3
"""A script that works on a DataFrame"""


import pandas as pd


def rename(df):
    """A function that performs operations on a DataFrame"""

    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]

    return df
