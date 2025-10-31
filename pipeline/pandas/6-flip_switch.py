#!/usr/bin/env python3
"""A function that takes a DataFrame and does sorting operations"""


def flip_switch(df):
    """A function that takes a DataFrame and does sorting operations"""
    sorted = df.sort_values(by='Timestamp', ascending=False)
    return sorted.T
