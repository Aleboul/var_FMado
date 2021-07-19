import numpy as np

from var_FMado.bivariate.base import Bivariate, CopulaTypes, Archimedean

class Clayton(Archimedean):
    """Class for clayton copula model"""

    copula_type = CopulaTypes.CLAYTON
    theta_interval = [-1,float('inf')] 
    invalid_thetas = [0]

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

class Amh(Archimedean):
    """Class for AMH copula model"""

    copula_type = CopulaTypes.AMH
    theta_interval = [-1.0,1.0] 
    invalid_thetas = [1.0]

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

class Frank(Archimedean):
    """Class for a frank copula model"""

    copula_type = CopulaTypes.FRANK
    theta_interval = [-float('inf'), float('inf')]
    invalid_thetas = [0]

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

class Nelsen_9(Archimedean):
    """Class for Nelsen_9 copula model"""

    copula_type = CopulaTypes.NELSEN_9
    theta_interval = [0,1] 
    invalid_thetas = [0]

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

class Nelsen_10(Archimedean):
    """Class for Nelsen_10 copula model"""

    copula_type = CopulaTypes.NELSEN_10
    theta_interval = [0,1] 
    invalid_thetas = [0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return np.log(2*np.power(t,-self.theta)-1)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        value_1 = np.exp(t) + 1
        value_2 = 2
        return(np.power(value_1 / value_2, -1/self.theta))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_1 = -self.theta*2*np.power(t,-1-self.theta)
        value_2 = 2*np.power(t,-self.theta)-1
        return(value_1 / value_2)

class Nelsen_11(Archimedean):
    """Class for Nelsen_10 copula model"""

    copula_type = CopulaTypes.NELSEN_10
    theta_interval = [0,0.5] 
    invalid_thetas = [0.5]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return np.log(2-np.power(t,self.theta))

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        return np.power(2-np.exp(t), 1/self.theta)

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_1 = -self.theta*np.power(t, self.theta-1)
        value_2 = 2 - np.power(t, self.theta)
        return(value_1 / value_2)

class Nelsen_12(Archimedean):
    """Class for Nelsen_12 copula model"""

    copula_type = CopulaTypes.NELSEN_12
    theta_interval = [0,float('inf')] 
    invalid_thetas = [0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return np.power(1/t - 1 , self.theta)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        value_1 = 1 + np.power(t, 1/self.theta)
        return(np.power(value_1,-1))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_ = self.theta*(-1/np.power(t,2))*np.power(1/t - 1, self.theta - 1)
        return(value_)

class Nelsen_13(Archimedean):
    """Class for Nelsen_13 copula model"""

    copula_type = CopulaTypes.NELSEN_13
    theta_interval = [0,float('inf')] 
    invalid_thetas = [0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return np.power(1-np.log(t),self.theta)-1

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        value_1 = 1-np.power(t+1,1/self.theta)
        return(np.exp(value_1))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_ = -self.theta*np.power(1-np.log(t),self.theta-1) / t
        return(value_)

class Nelsen_14(Archimedean):
    """Class for Nelsen_14 copula model"""

    copula_type = CopulaTypes.NELSEN_14
    theta_interval = [1.0,float('inf')] 
    invalid_thetas = [1.0]

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return np.power(np.power(t,-1/self.theta)-1, self.theta)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        return(np.power(np.power(t,1/self.theta)+1,-self.theta))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_1 = -np.power(t,-self.theta - 1)
        value_2 = np.power(t,-1/self.theta)-1
        return(value_1 * np.power(value_2,self.theta-1))

class Nelsen_15(Archimedean):
    """Class for Nelsen_15 copula model"""

    copula_type = CopulaTypes.NELSEN_15
    theta_interval = [1,float('inf')] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return np.power(1-np.power(t, 1/self.theta) , self.theta)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        return(np.power(1-np.power(t, 1/self.theta) , self.theta))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_ = - np.power(t, 1/self.theta - 1)*np.power(1-np.power(t, self.theta), self.theta - 1)
        return(value_)

class Nelsen_16(Archimedean):
    """Class for Nelsen_16 copula model"""

    copula_type = CopulaTypes.NELSEN_16
    theta_interval = [1,float('inf')] 
    invalid_thetas = []

    def _generator(self, t):
        """Return the generator function.
        .. math:: \phi(t) = 1/\theta(t^{-\theta}-1), \quad 0 < t < 1
        """
        return np.power(1-np.power(t, 1/self.theta) , self.theta)

    def _generator_inv(self, t):
        """Return the generator inverse.
        .. math:: (1-\theta) / (exp(t) - \theta), \quad t \geq 0
        """
        return(np.power(1-np.power(t, 1/self.theta) , self.theta))

    def _generator_dot(self, t):
        """Return the derivative of the generator function
        .. math:: \phi'(t) = \theta(-ln(t))^\theta / (t*ln(t))
        """
        value_ = - np.power(t, 1/self.theta - 1)*np.power(1-np.power(t, self.theta), self.theta - 1)
        return(value_)

class Nelsen_22(Archimedean):
    """Class for Nelsen_22 copula model"""

    copula_type = CopulaTypes.NELSEN_22
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
