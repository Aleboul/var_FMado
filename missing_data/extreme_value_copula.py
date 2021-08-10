import numpy as np
import math

from base import Bivariate, CopulaTypes, Extreme
from scipy.stats import norm
from scipy.stats import t

def gauss(x):

    return (1/math.pow(2*math.pi,1/2)) * math.exp(-math.pow(x,2)/2)

class Hussler_Reiss(Extreme):
    """Class for Hussler_Reiss copula model"""

    copula_type = CopulaTypes.HUSSLER_REISS
    theta_interval = [0,float('inf')] 
    invalid_thetas = []

    def _A(self, t):
        """Return the generator function.
        .. math:: A(t) = (1-t)\Phi(\theta + \frac{1}{2\theta}log\frac{1-t}{t}) + t\Phi(\theta + \frac{1}{2\theta}log\frac{t}{1-t}), \quad 0 < t < 1
        """
        value_1 = (1-t) * norm.cdf(self.theta + 1/(2*self.theta) * math.log((1-t)/t))
        value_2 = t * norm.cdf(self.theta + 1/(2*self.theta)*math.log(t/(1-t)))
        return value_1 + value_2

    def _Adot(self, t):
        """Return the derivative
        """
        value_1 = norm.cdf(self.theta + 1 / (2*self.theta) * math.log((1-t)/t))
        value_2 = (1/t) * norm.pdf(self.theta + 1 / (2*self.theta) * math.log((1-t)/t))
        value_3 = norm.cdf(self.theta + 1/(2*self.theta) * math.log(t/(1-t)))
        value_4 = (1/(1-t)) * norm.pdf(self.theta + 1/(2*self.theta) * math.log(t/(1-t)))
        return - value_1 - value_2 + value_3 + value_4

class Asy_log(Extreme):
    """Class for asymmetric logistic model copula model"""

    copula_type = CopulaTypes.ASYMMETRIC_LOGISTIC
    theta_interval = [0,float('inf')] 
    invalid_thetas = []

    def _A(self, t):
        """Return the Pickands dependence function.
        .. math:: \A(t) = (1-\psi_1)t + (1-\psi_2)(1-t) + ((\psi_1t)^\theta + (\psi_2(1-t))^\theta)^\frac{1}{\theta}, \quad 0 < t < 1
        """
        value_ = (1-self.psi1)*t+(1-self.psi2)*(1-t)+math.pow(math.pow(self.psi1*t,self.theta) + math.pow(self.psi2*(1-t), self.theta), 1/self.theta)
        return value_

    def _Adot(self, t):
        """Return the derivative of the Pickands dependence function.
        """
        value_1 = math.pow(self.psi1*t,self.theta)/t - math.pow(self.psi2*(1-t), self.theta)/(1-t)
        value_2 = math.pow(self.psi1*t,self.theta) + math.pow(self.psi2*(1-t), self.theta)
        return self.psi2 - self.psi1 + value_1 * math.pow(value_2, 1/(self.theta)-1)

class Asy_neg_log(Extreme):
    """Class for asymmetric negative logistic model copula model"""

    copula_type = CopulaTypes.ASYMMETRIC_NEGATIVE_LOGISTIC
    theta_interval = [1,float('inf')] 
    invalid_thetas = []

    def _A(self, t):
        """Return the Pickands dependence function.
        .. math:: \A(t) = 1-[(\psi_1(1-t))^{-\theta} + (\psi_2t)^{-\theta}]^\frac{1}{\theta}, \quad 0 < t < 1
        """
        value_ = 1-math.pow(math.pow(self.psi1*(1-t),-self.theta) + math.pow(self.psi2*t,-self.theta), -1/self.theta)
        return value_

    def _Adot(self, t):
        """Return the derivative of the Pickands dependence function.
        """
        value_1 = 1/((1-t)*math.pow(self.psi1*(1-t), self.theta)) - 1/(t*math.pow(self.psi2*t,self.theta))
        value_2 = math.pow(self.psi2*t, -self.theta) + math.pow(self.psi1*(1-t),-self.theta)
        return value_1*math.pow(value_2,-1/self.theta-1)

class Asy_mix(Extreme):
    """Class for asymmetric mixed model"""

    copula_type = CopulaTypes.ASSYMETRIC_MIXED_MODEL
    theta_interval = [0,float('inf')] 
    invalid_thetas = []

    def _A(self, t):
        """Return the Pickands dependence function.
        .. math:: \phi(t) = (-log(t))^\theta, \quad 0 < t < 1
        """
        value_ = 1-(self.theta + self.psi1)*t + self.theta*math.pow(t,2) + self.psi1*math.pow(t,3)
        return value_

    def _Adot(self, t):
        """Return the derivative of the Pickands dependence function.
        """
        value_ =-(self.theta+self.psi1) + 2*self.theta*t+3*self.psi1*math.pow(t,2)
        return value_

    def check_parameters(self):
        """
            Validate the parameters inserted

            This method is used to assert if the parameters are in the valid range for the copula

            Raises :
                ValueError : If theta or psi_1 does not satisfy the constraints.
        """

        if (not self.theta >= 0) or (not self.theta + 3*self.psi1 >=0) or (not self.theta + self.psi1 <= 1) or (self.theta + 2*self.psi1 <= 1):
            message = 'Parameters inserted {}, {} does not satisfy the inequalities for the given {} copula'
            raise ValueError(message.format(self.theta, self.psi1, self.copulaTypes.name))

class Student(Extreme):
    """Class for Student copula"""

    copula_type = CopulaTypes.STUDENT
    theta_interval = [-1,1] 
    invalid_thetas = []

    def z(self,w):
        value_ = math.pow((1+self.psi1),1/2)*(math.pow(w/(1-w),1/self.psi1) - self.theta)*math.pow(1-math.pow(self.theta,2),-1/2)
        return value_

    def _A(self, w):
        """Return the Pickands dependence function.
        .. math:: A(w) = wt_{\chi+1}(z_w)+(1-w)t_{\chi+1}(z_{1-w}) \quad 0 < w < 1
        .. math:: z_w  = (1+\chi)^\frac{1}{2}[(w/(1-w))^\frac{1}{\chi} - \rho](1-\rho^2)^\frac{-1}{2}
        """
        value_ = w*t.cdf(self.z(w), df = self.psi1 + 1)+(1-w)*t.cdf(self.z(1-w), df = self.psi1 + 1)
        return value_

    def _Adot(self, w):
        """Return the derivative of the Pickands dependence function.
        """
        value_1 = t.cdf(self.z(w), df = self.psi1 +1)
        value_2 = (1/(1-w)) * t.pdf(self.z(w), df = self.psi1+1)  * math.pow((1+self.psi1),1/2) * math.pow(1-math.pow(self.theta,2),-1/2) * math.pow(w/(1-w), 1/self.psi1)
        value_3 = t.cdf(self.z(1-w), df = self.psi1 + 1)
        value_4 = (1/w) * t.pdf(self.z(1-w), df = self.psi1 + 1) * math.pow((1+self.psi1),1/2) * math.pow(1-math.pow(self.theta,2),-1/2) * math.pow((1-w)/w, 1/self.psi1)
        return  value_1 + value_2 - value_3 - value_4
    