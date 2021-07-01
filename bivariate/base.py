import numpy as np
import math

from enum import Enum
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq

def min(a, b):
  
    if a <= b:
        return a
    else:
        return b

class CopulaTypes(Enum):
    """ Available copula families. """

    CLAYTON = 1
    AMH = 3
    GUMBEL = 4
    FRANK = 5
    JOE = 6
    NELSEN_9 = 9
    NELSEN_18 = 18
    NELSEN_22 = 22
    HUSSLER_REISS = 23
    ASYMMETRIC_LOGISTIC = 24
    ASYMMETRIC_NEGATIVE_LOGISTIC = 25
    ASSYMETRIC_MIXED_MODEL = 26
    STUDENT = 27

class Bivariate(object):
    """
        Base class for bivariate copulas.
        This class allows to instantiate all its subclasses and serves
        as a unique entry point for the bivariate copulas classes.
        It permit also to compute the variance of the FMadogram for a given point.
        Inputs
        ------
            copula_type : subtype of the copula
            random_seed : Seed for the random generator
        Attributes
        ----------
            copula_type(CopulaTypes) : Family of the copula a subclass belongs to
            theta_interval(list[float]) : Interval of vlid thetas for the given copula family
            invalid_thetas(list[float]) : Values that, even though they belong to
                :attr:`theta_interval`, shouldn't be considered valid.
            theta(float) : Parameter for the copula
            var_FMado(float) : value of the theoretical variance for a given point in [0,1]
    """
    copula_type = None
    _subclasses = []
    theta_interval = []
    invalid_thetas = []
    theta = []
    n_sample = []
    psi1 = []
    psi2 = []

    def __init__(self, copula_type = None, random_seed = None, theta = None, n_sample = None, psi1 = None, psi2 = None):
        """
            Initialize Bivariate object.
            Args:
            -----
                Copula_type (CopulaType or str) : subtype of the copula.
                random_seed (int or None) : Seed for the random generator
                theta (float or None) : Parameter for the copula.
        """
        self.random_seed = random_seed
        self.theta = theta
        self.n_sample = n_sample
        self.psi1 = psi1
        self.psi2 = psi2

    def chech_theta(self):
        """
            Validate the theta inserted
            Raises :
                ValueError : If thete is not in :attr:`theta_interval` or is in :attr:`invalid_thetas`,

        """
        lower, upper = self.theta_interval
        if (not lower <= self.theta <= upper) or (self.theta in self.invalid_thetas):
            message = 'The inserted theta value {} is out of limits for the given {} copula.'
            raise ValueError(message.format(self.theta, self.copula_type.name))

    def _integrand_v1(self,x,y,lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._C(min(u1,v1), min(u2,v2))
        value_2 = self._C(u1,u2)
        value_3 = self._C(v1,v2)
        value_  = value_1 - value_2 * value_3
        return(value_)

    def _integrand_v2(self,x,y,lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._dotC1(u1,u2)
        value_2 = self._dotC1(v1,v2)
        value_3 = (min(u1, v1) - u1*v1)
        value_  = value_1 * value_2 * value_3
        return(value_)
        
    def _integrand_v3(self,x,y,lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._dotC2(u1,u2)
        value_2 = self._dotC2(v1,v2)
        value_3 = (min(u2, v2) - u2*v2)
        value_  = value_1 * value_2 * value_3
        return(value_)

    def _integrand_cv12(self, x,y,lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._dotC1(v1,v2)
        value_2 = self._C(min(u1,v1), u2) - self._C(u1,u2) * v1
        value_  = value_1 * value_2
        return(value_)

    def _integrand_cv13(self, x,y, lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._dotC2(v1,v2)
        value_2 = self._C(u1, min(u2,v2)) - v2 * self._C(u1,u2)
        value_  = value_1 * value_2
        return(value_)

    def _integrand_cv23(self,x,y,lmbd):
        u1 = math.pow(x,1/lmbd)
        u2 = math.pow(x,1/(1-lmbd))
        v1 = math.pow(y,1/(lmbd))
        v2 = math.pow(y,1/(1-lmbd))
        value_1 = self._dotC1(u1,u2)
        value_2 = self._dotC2(v1,v2)
        value_3 = self._C(u1, v2) - u1 * v2
        value_  = value_1 * value_2 * value_3
        return(value_)

    def var_FMado(self,lmbd):
        v1 = dblquad(lambda x,y : self._integrand_v1(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        v2 = dblquad(lambda x,y : self._integrand_v2(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        v3 = dblquad(lambda x,y : self._integrand_v3(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        cv12 = dblquad(lambda x,y : self._integrand_cv12(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        cv13 = dblquad(lambda x,y : self._integrand_cv13(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        cv23 = dblquad(lambda x,y : self._integrand_cv23(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        return(v1 + v2 + v3 - 2*cv12 - 2*cv13 + 2 * cv23)

class Archimedean(Bivariate):
    """
        Base class for bivariate archimedean copulas.
        This class allows to use methods which use the generator 
        function and its inverse.
        
        Inputs
        ------

        Attributes
        ----------
            sample_uni (np.array[float]) : sample where the margins are uniform on [0,1]
            sample (np.array[float]) : sample where the uniform margins where inverted by
                                        a generalized inverse of a quantile function.
    """

    def _C(self,u,v):
        """Return the value of the copula taken on (u,v)
        .. math:: C(u,v) = \phi^\leftarrow (\phi(u) + \phi(v)), \quad 0<u,v<1
        """
        value_ = self._generator_inv(self._generator(u) + self._generator(v))
        return value_
    def _dotC1(self,u,v):
        """Return the value of the first partial derivative taken on (u,v)
        .. math:: C(u,v) = \phi'(u) / \phi'(C(u,v)), \quad 0<u,v<1
        """ 
        value_1 = self._generator_dot(u) 
        value_2 = self._generator_dot(self._C(u,v))
        return value_1 / value_2

    def _dotC2(self,u,v):
        """Return the value of the first partial derivative taken on (u,v)
        .. math:: C(u,v) = \phi'(v) / \phi'(C(u,v)), \quad 0<u,v<1
        """
        value_1 = self._generator_dot(v) 
        value_2 = self._generator_dot(self._C(u,v))
        return value_1 / value_2

    def _generate_randomness(self):
        """
            Generate a bivariate sample draw identically and
            independently from a uniform over the segment [0,1]
            Inputs
            ------
            n_sample : length of the bivariate sample
            Outputs
            -------
            n_sample x 2 np.array
        """

        v_1 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample) # first sample
        v_2 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample) # second sample
        output_ = np.vstack([v_1, v_2]).T
        return output_

    def sample_unimargin(self):
        """
            Draws a bivariate sample from archimedean copula
            Margins are uniform
        """
        output = np.zeros((self.n_sample,2))
        X = self._generate_randomness()
        for i in range(0,self.n_sample):
            v = X[i]
            def func(x):
                value_ = np.abs(( x - self._generator(x) / self._generator_dot(x)) - v[1])
                return(value_)
            sol = minimize_scalar(func, bounds = (0,1), method = "bounded")
            sol = float(sol.x)
            u = [self._generator_inv(v[0] * self._generator(sol)) , self._generator_inv((1-v[0])*self._generator(sol))]
            output[i,:] = u
        return output

    def sample(self):
        """
        
        """
        intput = self.sample_unimargin()
        output = np.zeros((self.n_sample,2))
        ncol = intput.shape[1]
        for i in range(0, ncol):
            output[:,i] = norm.ppf(intput[:,i])

        return (output)

class Extreme(Bivariate):
    """
        Base class for extreme value copulas.
        This class allows to use methods which use the Pickans dependence function.
        
        Inputs
        ------

        Attributes
        ----------
            sample_uni (np.array[float]) : sample where the margins are uniform on [0,1]
            sample (np.array[float]) : sample where the uniform margins where inverted by
                                        a generalized inverse of a quantile function.
    """

    def _C(self,u,v):
        """Return the value of the copula taken on (u,v)
        .. math:: C(u,v) = (uv)^{A(\frac{log(v)}{log(uv)})}, \quad 0<u,v<1
        """
        value_ = math.pow(u*v, self._A(math.log(v) / math.log(u*v)))
        return value_

    def _Kappa(self,t):
        """Return the value of Kappa taken on t
        .. math:: Kappa(t) = A(t) - t*A'(t), \quad 0<t<1
        """
        return (self._A(t) - t * self._Adot(t))

    def _Zeta(self, t):
        """Return the value of Zeta taken on t
        .. math:: Kappa(t) = A(t) + (1-t)*A'(t), \quad 0<t<1
        """
        return (self._A(t) + (1-t)*self._Adot(t))

    def _dotC1(self,u,v):
        """Return the value of \dot{C}_1 taken on (u,v)
        .. math:: \dot{C}_1  = (C(u,v) / u) * (Kappa(log(v)/log(uv))), \quad 0<u,v<1
        """
        t = math.log(v) / math.log(u*v)
        value_ = (self._C(u,v) / u) * self._Kappa(t)
        return(value_)

    def _dotC2(self, u,v):
        """Return the value of \dot{C}_2 taken on (u,v)
        .. math:: \dot{C}_2  = (C(u,v) / v) * (Zeta(log(v)/log(uv))), \quad 0<u,v<1
        """
        t = math.log(v) / math.log(u*v)
        value_ = (self._C(u,v) / v) * self._Zeta(t)
        return(value_)

    def _generate_randomness(self):
        """
            Generate a bivariate sample draw identically and
            independently from a uniform over the segment [0,1]
            Inputs
            ------
            n_sample : length of the bivariate sample
            Outputs
            -------
            n_sample x 2 np.array
        """

        v_1 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample) # first sample
        v_2 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample) # second sample
        output_ = np.vstack([v_1, v_2]).T
        return output_

    def sample_unimargin(self):
        """
            Draws a bivariate sample from archimedean copula
            Margins are uniform
        """
        Epsilon = 1e-12
        output = np.zeros((self.n_sample,2))
        X = self._generate_randomness()
        for i in range(0,self.n_sample):
            v = X[i]
            def func(x):
                value_ = self._dotC1(v[0],x) - v[1]
                return(value_)
            sol = brentq(func, Epsilon,1-Epsilon)
            print(func(sol))
            u = [v[0], sol]
            output[i,:] = u
        return output