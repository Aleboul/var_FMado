import numpy as np

from base import Bivariate, CopulaTypes, Archimedean

class Amh(Archimedean):
    """Class for AMH copula model"""

    copula_type = CopulaTypes.AMH
    theta_interval = [-1,1] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return np.log((1-self.theta*(1-t)) / t)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        value_1 = 1-self.theta
        value_2 = np.exp(t) - self.theta
        value_  = value_1 / value_2
        return(value_)

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_1 = self.theta-1
        value_2 = t*(1-self.theta*(1-t))
        return(value_1 / value_2)
