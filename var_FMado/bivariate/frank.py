import numpy as np

from base import Bivariate, CopulaTypes, Archimedean

class Frank(Archimedean):
    """Class for a frank copula model"""

    copula_type = CopulaTypes.FRANK
    theta_interval = [] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return -np.log((np.exp(-self.theta*t)-1) / (np.exp(-self.theta)-1))

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        return - (1 / self.theta)*np.log(1+np.exp(-t)*(np.exp(-self.theta)-1))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_1 = self.theta * np.exp(-self.theta * t)
        value_2 = np.exp(-self.theta*t) - 1
        return(value_1 / value_2)
