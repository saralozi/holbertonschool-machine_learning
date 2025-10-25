#!/usr/bin/env python3
"""A script that creates a class to represent a binomial distribution"""


class Binomial:
    """Binomial Distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p > 1 or p < 0:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data)/len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            prob = float(1 - (variance / mean))
            num = round(mean / prob)
            prob = mean / num

            self.n = int(num)
            self.p = float(prob)
