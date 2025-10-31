#!/usr/bin/env python3
"""A script that takes a DatFrame and handles missing data"""


def fill(df):
    """A function that handles missing data"""

    df = df.drop('Weighted_Price', axis=1)

    df['Close'] = df['Close'].fillna(method='ffill', inplace=True)

    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'], inplace=True)

    df[['Volume_(BTC)', 'Volume_(Currency)']] = df[
        ['Volume_(BTC)', 'Volume_(Currency)']].fillna(0, inplace=True)

    return df
