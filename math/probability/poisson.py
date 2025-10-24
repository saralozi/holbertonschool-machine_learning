#!/usr/bin/env python3
"""A script that creates a class to represent a poission distribution"""


class Poisson:
    """Poisson Distribution"""

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
            self.lambtha = float((sum(data))/len(data))

    def pmf(self, k):
        """A function to calculate PMF"""

        if not isinstance(k, int):
            int(k)
        if k < 0:
            return 0
        lambtha = self.lambtha
        fact = 1
        for i in range(1, k + 1):
            fact *= i
        return ((2.7182818285**-lambtha)*(lambtha**k))/fact
