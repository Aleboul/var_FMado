import numpy as np

from base import Bivariate, CopulaTypes, Archimedean

class Joe(Archimedean):
    """Class for clayton copula model"""

    copula_type = CopulaTypes.JOE
    theta_interval = [1,float('inf')] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return -np.log(1-np.power(1-t, self.theta))

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        return 1 - np.power((1-np.exp(-t)), 1/self.theta)

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_1 = -self.theta * np.power(1-t, self.theta-1)
        value_2 = 1 - np.power(1-t, self.theta) 
        return(value_1 / value_2)
