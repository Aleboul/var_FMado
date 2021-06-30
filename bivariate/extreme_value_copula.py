import numpy as np
import math

from var_FMado.bivariate.base import Bivariate, CopulaTypes, Extreme
from scipy.stats import norm

def gauss(x):
    return (1/math.pow(2*math.pi,1/2)) * math.exp(-math.pow(x,2)/2)

class Gumbel(Extreme):
    """Class for clayton copula model"""

    copula_type = CopulaTypes.GUMBEL
    theta_interval = [1,float('inf')] 
    invalid_thetas = []

    def _A(self, t):
        """Return the generator function.
        .. math:: \phi(t) = (-log(t))^\theta, \quad 0 < t < 1
        """
        value_ = math.pow(t, self.theta) + math.pow(1-t, self.theta)
        return math.pow(value_, 1/self.theta)

    def _Adot(self, t):
        """Return the generator inverse.
        .. math:: \phi^\leftarrow(u) = e^{-t^{1/\theta}}, \quad t \geq 0
        """
        value_1 = math.pow(t, self.theta-1) - math.pow(1-t, self.theta-1)
        value_2 = math.pow(t, self.theta) + math.pow(1-t, self.theta)
        return value_1 * math.pow(value_2, (1/self.theta)-1)

class Hussler_Reiss(Extreme):
    """Class for clayton copula model"""

    copula_type = CopulaTypes.HUSSLER_REISS
    theta_interval = [0,float('inf')] 
    invalid_thetas = []

    def _A(self, t):
        """Return the generator function.
        .. math:: \phi(t) = (-log(t))^\theta, \quad 0 < t < 1
        """
        value_1 = (1-t) * norm.cdf(self.theta + 1/(2*self.theta) * math.log((1-t)/t))
        value_2 = t * norm.cdf(self.theta + 1/(2*self.theta)*math.log(t/(1-t)))
        return value_1 + value_2

    def _Adot(self, t):
        """Return the generator inverse.
        .. math:: \phi^\leftarrow(u) = e^{-t^{1/\theta}}, \quad t \geq 0
        """
        value_1 = norm.cdf(self.theta + 1 / (2*self.theta) * math.log((1-t)/t))
        value_2 = (1/t) * gauss(self.theta + 1 / (2*self.theta) * math.log((1-t)/t))
        value_3 = norm.cdf(self.theta + (1/2*self.theta) * math.log(t/(1-t)))
        value_4 = (1/(1-t)) * gauss(self.theta + (1/2*self.theta) * math.log(t/(1-t)))
        return - value_1 - value_2 + value_3 + value_4