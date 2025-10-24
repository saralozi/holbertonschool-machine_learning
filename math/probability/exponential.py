#!/usr/bin/env python3
"""A script that creates a class to represent an exponential distribution"""


class Exponential:
    """Exponential Distribution"""

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = 0
            mean = sum(data)/len(data)
            self.lambtha = float(1/mean)

    def pdf(self, x):
        """A function to calculate PDF"""
        lambtha = self.lambtha

        if x < 0:
            return 0
        else: 
            return (lambtha*((2.7182818285)**(-lambtha*x)))
