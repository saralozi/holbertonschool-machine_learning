#!/usr/bin/env python3
"""A function that calculates the weighted moving average of a data set"""


def moving_average(data, beta):
    """Compute exponential moving average"""

    moving_avg = []
    v = 0

    for t, value in enumerate(data, start=1):

        v = beta * v + (1 - beta) * value
        v_corrected = v / (1 - beta ** t)

        moving_avg.append(v_corrected)

    return moving_avg
