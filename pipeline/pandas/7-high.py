#!/usr/bin/env python3
"""A function that takes a DataFrame and does sorting operations"""


def high(df):
    """A function that does sorting operations"""
    sorted = df.sort_values(by='High', ascending=False)
    return sorted
