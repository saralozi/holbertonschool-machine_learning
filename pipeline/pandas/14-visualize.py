#!/usr/bin/env python3
"""A script that visualizes a DataFrame"""


import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop('Weighted_Price', axis=1)

df = df.rename(columns={'Timestamp': 'Date'})

df['Date'] = pd.to_datetime(df['Date'], unit='s')

df = df.set_index('Date')

df['Close'] = df['Close'].fillna(method='ffill')

df[['High', 'Low', 'Open']] = df[['High', 'Low', 'Open']].fillna(df['Close'])

df[['Volume_(BTC)', 'Volume_(Currency)']] = df[
    ['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)

df_2017_and_later = df.loc['2017-01-01':]

df_2017_and_later.resample('D')

daily = df_2017_and_later.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

print(daily)
