import numpy as np

from base import Bivariate, CopulaTypes, Archimedean

class Nelsen_18(Archimedean):
    """Class for Nelsen_18 copula model"""

    copula_type = CopulaTypes.NELSEN_18
    theta_interval = [2,float('inf')] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return np.exp(self.theta / (t-1))

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        return( (self.theta / np.log(t)) + 1)

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_1 = - self.theta * np.exp(self.theta / (t-1))
        value_2 = np.power(t-1,2)
        return(value_1 / value_2)

class Nelsen_22(Archimedean):
    """Class for Nelsen_22 copula model"""

    copula_type = CopulaTypes.NELSEN_18
    theta_interval = [0,1] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return np.arcsin(1-np.power(t,self.theta))

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        return(np.power(1-np.sin(t),1/self.theta))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_1 = - self.theta * np.power(t, self.theta-1)
        value_2 = np.power(1-np.power(np.power(t,self.theta)-1,2), 1/2)
        return(value_1 / value_2)

class Nelsen_9(Archimedean):
    """Class for Nelsen_9 copula model"""

    copula_type = CopulaTypes.NELSEN_9
    theta_interval = [0,1] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return np.log(1-self.theta*np.log(t))

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        return(np.exp((1-np.exp(t))/self.theta))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_1 = self.theta
        value_2 = t * (self.theta * np.log(t) - 1)
        return(value_1 / value_2)
