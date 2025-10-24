#!/usr/bin/env python3
"""A script that creates a class to represent a normal distribution"""


class Normal:
    """Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
            return

        if not isinstance(data, list):
            raise TypeError("data must be a list")
        if len(data) < 2:
            raise ValueError("data must contain multiple values")

        mean = sum(data) / len(data)
        stddev = (sum((mean - num) ** 2 for num in data) / len(data)) ** 0.5

        self.mean = float(mean)
        self.stddev = float(stddev)

    def z_score(self, x):
        """A function that calculates the z-score of a given x-value"""

        return ((x-self.mean)/self.stddev)

    def x_value(self, z):
        """A function that calculates the x-value of a given z-score"""

        return (self.mean+(z*self.stddev))

    def pdf(self, x):
        """A function that calculates the value
        of the PDF for a given x-value"""

        return ((1/(self.stddev*((2*3.1415926536)**0.5))) *
                (2.7182818285)**(-0.5*(((x-self.mean)/self.stddev)**2)))

    def cdf(self, x):
        """A function that calculates the value
        of the CDF for a given x-value"""

        z = (x-self.mean)/(self.stddev*(2**0.5))
        erf = (2/(3.1415926536**0.5))*(
            z-(z**3)/3+(z**5)/10-(z**7)/42+(z**9)/216)
        return 0.5 * (1 + erf)
