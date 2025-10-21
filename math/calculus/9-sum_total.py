#!/usr/bin/env python3

"""A script that calculates the sum of powers of n"""


def summation_i_squared(n):
    """A function that calculates the sum of powers of n"""
    if type(n) is not int:
        return None
    else:
        return int((n*(n+1)*(2*n+1))//6)
