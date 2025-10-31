#!/usr/bin/env python 3
"""A script that takes 2 DataFrames and does operations"""


import pandas as pd


def hierarchy(df1, df2):
    """A function that takes 2 DataFrames and does operations"""

    index = __import__('10-index').index
    df1 = index(df1)
    df2 = index(df2)

    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    new_df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    new_df.index = new_df.index.set_names(["Exchange", "Timestamp"])
    new_df = new_df.swaplevel(0, 1)

    new_df = new_df.sort_index(level="Timestamp")

    return new_df
