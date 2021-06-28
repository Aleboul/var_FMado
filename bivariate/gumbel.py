import numpy as np

from base import Bivariate, CopulaTypes, Archimedean

class Gumbel(Archimedean):

    copula_type = CopulaTypes.GUMBEL
    theta_interval = [1,float('inf')]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = (-log(t))^\theta, \quad 0 < t < 1
        """
        return np.power(-np.log(t), self.theta)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: \phi^\leftarrow(u) = e^{-t^{1/\theta}}, \quad t \geq 0
        """
        return np.exp(-np.power(t,1/self.theta))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_1 = self.theta*np.power((-np.log(t)), self.theta)
        value_2 = t * np.log(t)
        value_  = value_1 / value_2
        return(value_)
