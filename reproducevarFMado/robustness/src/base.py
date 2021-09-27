import numpy as np
import math

from enum import Enum
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq, minimize_scalar

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
    NELSEN_10 = 10
    NELSEN_12 = 12
    NELSEN_13 = 13
    NELSEN_14 = 14
    NELSEN_15 = 15
    NELSEN_16 = 16
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
            sample (np.array[float]) : sample where the uniform margins are inverted by
                                        a generalized inverse of a cdf.
    """
    copula_type = None
    theta_interval = []
    invalid_thetas = []
    theta = []
    n_sample = []
    psi1 = []
    psi2 = []

    def __init__(self, random_seed = None, theta = None, n_sample = None, psi1 = None, psi2 = None):
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

    def check_theta(self):
        """
            Validate the theta inserted
            Raises :
                ValueError : If there is not in :attr:`theta_interval` or is in :attr:`invalid_thetas`,

        """
        lower, upper = self.theta_interval
        if (not lower <= self.theta <= upper) or (self.theta in self.invalid_thetas):
            message = 'The inserted theta value {} is out of limits for the given {} copula.'
            raise ValueError(message.format(self.theta, self.copula_type.name))

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
        np.random.seed(self.random_seed)
        v_1 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample)
        v_2 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample)
        output_ = np.vstack([v_1, v_2]).T
        return output_

    def sample(self, inv_cdf):
        """
            Draws a bivariate sample from archimedean copula and invert it by
            a given generalized inverse of cumulative distribution function
            Inputs
            ------
            inv_cdf : generalized inverse of cumulative distribution function
            Outputs
            -------
            n_sample x 2 np.array
        """
        intput = self.sample_unimargin()
        output = np.zeros((self.n_sample,2))
        ncol = intput.shape[1]
        for i in range(0, ncol):
            output[:,i] = inv_cdf(intput[:,i])

        return (output)

class Archimedean(Bivariate):
    """
        Base class for bivariate archimedean copulas.
        This class allows to use methods which use the generator 
        function and its inverse.
        
        Inputs
        ------

        Attributes
        ----------
            sample_uni (np.array[float]) : sample where the margins are uniform on [0,1].
            var_FMado ([float]) : give the asymptotic variance of the lambda-FMadogram 
                                  for an archimedean copula.
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

    def sample_unimargin(self):
        """
            Draws a bivariate sample from archimedean copula
            Margins are uniform
        """
        self.check_theta()
        output = np.zeros((self.n_sample,2))
        X = self._generate_randomness()
        for i in range(0,self.n_sample):
            v = X[i]
            #def func(x):
            #    value_ = ( x - self._generator(x) / self._generator_dot(x)) - v[1]
            #    return(value_)
            def func(x):
                value_ = np.abs((x - self._generator(x) / self._generator_dot(x)) - v[1])
                return value_
            sol = minimize_scalar(func, bounds = (0.0,1.0), method = 'bounded').x
            u = [self._generator_inv(v[0] * self._generator(sol)) , self._generator_inv((1-v[0])*self._generator(sol))]
            output[i,:] = u
        return output

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
        """
            Compute the asymptotic variance of the lambda-FMadogram
        """
        v1 = dblquad(lambda x,y : self._integrand_v1(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        v2 = dblquad(lambda x,y : self._integrand_v2(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        v3 = dblquad(lambda x,y : self._integrand_v3(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        cv12 = dblquad(lambda x,y : self._integrand_cv12(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        cv13 = dblquad(lambda x,y : self._integrand_cv13(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        cv23 = dblquad(lambda x,y : self._integrand_cv23(x,y, lmbd), 0,1.0, 0, 1.0)[0]
        return(v1 + v2 + v3 - 2*cv12 - 2*cv13 + 2 * cv23)

class Extreme(Bivariate):
    """
        Base class for extreme value copulas.
        This class allows to use methods which use the Pickands dependence function.
        
        Inputs
        ------

        Attributes
        ----------
            sample_uni (np.array[float]) : sample where the margins are uniform on [0,1].
            var_FMado ([float]) : give the asymptotic variance of the lambda-FMadogram
                                  for an extreme value copula.
    """

    def _C(self,u,v):
        """Return the value of the copula taken on (u,v)
        .. math:: C(u,v) = (uv)^{A(\frac{log(v)}{log(uv)})}, \quad 0<u,v<1
        """
        value_ = math.pow(u*v, self._A(math.log(v) / math.log(u*v)))
        return value_

    def _f(self, lmbd):
        return math.pow(lmbd*(1-lmbd)/(self._A(lmbd)+ lmbd*(1-lmbd)),2)

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
            Draw a bivariate sample from archimedean copula
            Margins are uniform
        """
        self.check_theta()
        Epsilon = 1e-12
        output = np.zeros((self.n_sample,2))
        X = self._generate_randomness()
        for i in range(0,self.n_sample):
            v = X[i]
            def func(x):
                value_ = self._dotC1(v[0],x) - v[1]
                return(value_)
            sol = brentq(func, Epsilon,1-Epsilon)
            u = [v[0], sol]
            output[i,:] = u
        return output

    def _integrand_ev1(self,s, lmbd):
        value_ = self._A(s) + (1-s)*(self._A(lmbd)/(1-lmbd) - (1-lmbd) -1) -s*lmbd+1
        return math.pow(value_,-2)
    def _integrand_ev2(self, s, lmbd):
        value_ = self._A(s)+s*(self._A(lmbd)/lmbd - lmbd -1) - (1-s)*(1-lmbd)+1
        return math.pow(value_,-2)

    def _integrand_ev3(self, s, lmbd):
        value_ = self._A(s)+(1-s)*(self._A(lmbd)/(1-lmbd) - (1-lmbd)-1) + s*(self._A(lmbd)/lmbd - lmbd - 1) + 1
        return math.pow(value_, -2)

    def true_FMado(self):
        value_ = 0.5 - 1/(2*self._A(0.5)+1)
        return value_

    def var_FMado(self, lmbd):
        """
            Compute asymptotic variance of lambda-FMadogram using the specific form for
            bivariate extreme value copula.
        """
        value_11 = self._A(lmbd) / (self._A(lmbd) + 2*lmbd*(1-lmbd))
        value_12 = (math.pow(self._Kappa(lmbd),2) * (1-lmbd)) / (2*self._A(lmbd) - (1-lmbd) + 2*lmbd*(1-lmbd))
        value_13 = (math.pow(self._Zeta(lmbd),2) * lmbd) / (2*self._A(lmbd) - lmbd + 2*lmbd*(1-lmbd))
        value_1  = self._f(lmbd) * (value_11 + value_12 + value_13)

        value_21 = (math.pow(1-lmbd,2) - self._A(lmbd)) / (2*self._A(lmbd) - (1-lmbd)+2*lmbd*(1-lmbd))
        value_22 = quad(lambda s : self._integrand_ev1(s, lmbd), 0.0, lmbd)[0]
        value_2  = self._Kappa(lmbd) * self._f(lmbd) * value_21 + self._Kappa(lmbd)*lmbd*(1-lmbd)*value_22

        value_31 = (math.pow(lmbd,2)-self._A(lmbd)) / (2*self._A(lmbd) - lmbd + 2*lmbd*(1-lmbd))
        value_32 = quad(lambda s : self._integrand_ev2(s,lmbd),lmbd,1.0)[0]
        value_3  = self._Zeta(lmbd) * self._f(lmbd) * value_31 + self._Zeta(lmbd) * lmbd * (1-lmbd)*value_32

        value_41 = self._f(lmbd)*self._Kappa(lmbd)*self._Zeta(lmbd)
        value_42 = quad(lambda s : self._integrand_ev3(s, lmbd), 0.0, 1.0)[0]
        value_4  = -value_41 + self._Kappa(lmbd) * self._Zeta(lmbd) * lmbd * (1-lmbd) * value_42

        value_  = value_1 - 2 * value_2 - 2 * value_3 + 2 * value_4
        return value_