import numpy as np

from base import Bivariate, CopulaTypes, Archimedean

class Clayton(Archimedean):
    """Class for clayton copula model"""

    copula_type = CopulaTypes.CLAYTON
    theta_interval = [-1,float('inf')] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = (-log(t))^\theta, \quad 0 < t < 1
        """
        return (1.0/self.theta)*(np.power(t, -self.theta) - 1)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: \phi^\leftarrow(u) = e^{-t^{1/\theta}}, \quad t \geq 0
        """
        return np.power((1.0+self.theta*t),-1/self.theta)

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_ = - np.power(t, -self.theta-1)
        return(value_)
