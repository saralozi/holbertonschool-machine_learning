#!/usr/bin/env python3
"""A script that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """A function that calculates the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0 or not isinstance(C, int):
        return None

    integral = [C]
    for i in range(len(poly)):
        res = poly[i] / (i + 1)
        res = int(res) if res.is_integer() else res
        integral.append(res)
    return integral
