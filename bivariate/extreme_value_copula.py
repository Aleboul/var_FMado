import numpy as np
import math

from var_FMado.bivariate.base import Bivariate, CopulaTypes, Extreme
from scipy.stats import norm
from scipy.stats import t

def gauss(x):

    return (1/math.pow(2*math.pi,1/2)) * math.exp(-math.pow(x,2)/2)

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

class Asy_log(Extreme):
    """Class for asymmetric logistic model copula model"""

    copula_type = CopulaTypes.ASYMMETRIC_LOGISTIC
    theta_interval = [1,float('inf')] 
    invalid_thetas = []

    def _A(self, t):
        """Return the Pickands dependence function.
        .. math:: \phi(t) = (-log(t))^\theta, \quad 0 < t < 1
        """
        value_ = (1-self.psi1)*(1-t)+(1-self.psi2)*t+math.pow(math.pow(self.psi1*t,self.theta) + math.pow(self.psi2*(1-t), self.theta), 1/self.theta)
        return value_

    def _Adot(self, t):
        """Return the derivative of the Pickands dependence function.
        .. math:: \phi^\leftarrow(u) = e^{-t^{1/\theta}}, \quad t \geq 0
        """
        value_1 = math.pow(self.psi1*t,self.theta)/t - math.pow(self.psi2*(1-t), self.theta)/(1-t)
        value_2 = math.pow(self.psi1*t,self.theta) + math.pow(self.psi2*(1-t), self.theta)
        return self.psi1 - self.psi2 + value_1 * math.pow(value_2, 1/(self.theta)-1)

class Asy_neg_log(Extreme):
    """Class for asymmetric negative logistic model copula model"""

    copula_type = CopulaTypes.ASYMMETRIC_NEGATIVE_LOGISTIC
    theta_interval = [1,float('inf')] 
    invalid_thetas = []

    def _A(self, t):
        """Return the Pickands dependence function.
        .. math:: \phi(t) = (-log(t))^\theta, \quad 0 < t < 1
        """
        value_ = 1-math.pow(math.pow(self.psi1*(1-t),-self.theta) + math.pow(self.psi2*t,-self.theta), -1/self.theta)
        return value_

    def _Adot(self, t):
        """Return the derivative of the Pickands dependence function.
        .. math:: \phi^\leftarrow(u) = e^{-t^{1/\theta}}, \quad t \geq 0
        """
        value_1 = 1/((1-t)*math.pow(self.psi1*(1-t), self.theta)) - 1/(t*math.pow(self.psi2*t,self.theta))
        value_2 = math.pow(self.psi2*t, -self.theta) + math.pow(self.psi1*(1-t),-self.theta)
        return value_1*math.pow(value_2,-1/self.theta-1)

class Asy_mix(Extreme):
    """Class for asymmetric mixed model"""

    copula_type = CopulaTypes.ASSYMETRIC_MIXED_MODEL
    theta_interval = [1,float('inf')] 
    invalid_thetas = []

    def _A(self, t):
        """Return the Pickands dependence function.
        .. math:: \phi(t) = (-log(t))^\theta, \quad 0 < t < 1
        """
        value_ = 1-(self.theta + self.psi1)*t + self.theta*math.pow(t,2) + self.psi1*math.pow(t,3)
        return value_

    def _Adot(self, t):
        """Return the derivative of the Pickands dependence function.
        .. math:: \phi^\leftarrow(u) = e^{-t^{1/\theta}}, \quad t \geq 0
        """
        value_ =-(self.theta+self.psi1) + 2*self.theta*t+3*self.psi1*math.pow(t,2)
        return value_

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
        .. math:: \phi(t) = (-log(t))^\theta, \quad 0 < t < 1
        """
        value_ = w*t.cdf(self.z(w), df = self.psi1 + 1)+(1-w)*t.cdf(self.z(1-w), df = self.psi1 + 1)
        return value_

    def _Adot(self, w):
        """Return the derivative of the Pickands dependence function.
        .. math:: \phi^\leftarrow(u) = e^{-t^{1/\theta}}, \quad t \geq 0
        """
        value_1 = math.pow((1+self.psi1),1/2)*(1/self.psi1)*(1/(w*(1-w)))*math.pow(w/(1-w), self.psi1)*math.pow((1-math.pow(self.theta,2)),-1/2)
        value_2 = -math.pow((1+self.psi1),1/2)*(1/self.psi1)*(1/(w*(1-w)))*math.pow((1-w)/w, self.psi1)*math.pow((1-math.pow(self.theta,2)),-1/2)
        value_  = t.cdf(self.z(w), df = self.psi1 + 1) + w*t.pdf(self.z(w), df = self.psi1) * value_1 - t.cdf(self.z(1-w), df = self.psi1+1) + (1-w)*t.pdf(self.z(1-w), df=self.psi1+1)*value_2 
        return value_
    