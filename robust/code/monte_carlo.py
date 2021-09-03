import numpy as np
import pandas as pd
from tqdm import tqdm

class Monte_Carlo(object):
    """
        Base class for Monte-Carlo simulations
        Inputs
        ------
            copula_sane : law of the bivariate sane random vector
            copula_outlier : law of the bivariate outlier random vector
            n_iter (int): number of Monte Carlo simulation
            n_sample (list of int or [int]): multiple length of sample
            lmbd (float) : parameter for the lambda-FMadogram
            random_seed (Union[int, None]) : seed for the random generator
            n_sample (list[int]) : several lengths used for estimation
            K (int) : number of block
            delta (float) : corruption of data, if delta = 0.5, no data is corrupted, if delta = 0, half of data are corrupted
        Attributes
        ----------
            lmbd_interval ([lower, upper]) : interval of valid thetas for the given lambda-FMadogram
            simu (DataFrame) : results of the Monte-Carlo simulation
                - FMado : the value of estimator
                - n : length of the sample
                - gp : number of iteration
                - scaled : ::math:: \sqrt{T}(\hat{\nu}_T (\lambda) - \nu)
    """

    n_iter = None
    n_sample = []
    lmbd = None
    lmbd_interval = []
    random_seed = None
    copula = None
    delta_interval = [0,0.5]

    def __init__(self, n_iter = None, n_sample = [], lmbd = 0.5, random_seed = None, copula_sane = None, copula_contaminated = None, K = None, delta = None):
        """
            Initialize Monte_Carlo object
        """
        self.n_iter = n_iter
        self.n_sample = n_sample
        self.lmbd = lmbd
        self.copula_sane = copula_sane
        self.copula_contaminated = copula_contaminated
        self.K = K
        self.delta = delta

    def check_lmbd(self):
        """
            Validate the lmbd
            This method is used to assert the lmbd insert by the user.
            Raises :
                ValueError : If lmbd is not in :attr:`lmbd_interval`
        """

        lower, upper = self.lmbd_interval
        if (not lower <= self.lmbd <= upper):
            message = "The lmbd value {} is out of limits for the given simulation."
            raise ValueError(message.format(self.lmbd))
        lower, upper = self.delta_interval
        if (not lower < self.delta <= upper):
            message = "The delta value {} is out of limits for the given simulation."
            raise ValueError(message.format(self.delta))

    def _ecdf(self,data):
        """
            Compute ECDF of the data
            Inputs
            ------
            data : array of data
            Outputs
            -------
            Empirical cumulative distribution function
        """

        index = np.argsort(data)
        ecdf = np.zeros(len(index))
        for i in index:
            ecdf[i] = (1.0 / len(index)) * np.sum(data <= data[i])
        return ecdf

    def _fmado(self, X):
        """
		    This function computes the Madogram
    
		    Inputs
		    ------
		    X : a matrix
		    lmbd : constant between 0 and 1 use for the lmbd F-madogram
		    Outputs
		    -------
		    A matrix equals to 0 if i = j and equals to |F(X)^{\lmbd} - G(Y)^{1-\lmbd}|
	    """

        Nnb = X.shape[1]
        Tnb = X.shape[0]

        V = np.zeros([Tnb, Nnb])
        for p in range(0, Nnb):
            X_vec = np.array(X[:,p])
            Femp = self._ecdf(X_vec)
            V[:,p] = Femp
        Fmado = np.linalg.norm(V[:,0] - V[:,1]) / (2 * Tnb)

        return Fmado

    def _fmado_MoN(self, X):
        """
            Compute the Median of meaNs with K blocks

            Inputs
            ------
            X (array) : array of observation
            K (int)   : number of blocks, sould divide the number of observations, throw away some observations if not

            Outputs
            -------
            \hat{nu}_{MoN}
        """
        sample_ = np.random.choice(X, len(X))
        n = len(sample_)
        if n % K != 0:
            N = (n//self.K) * self.K
            sample_ = sample_[0:N]
        sample_ = np.split(sample_, self.K)
        F_MoN = np.median([_fmado(sample_[k]) for k in range(0,K)], axis = 0)
        return F_MoN
        
    def huber_contamination(self, inv_cdf_s, inv_cdf_o, show_index = False):
        """
            Simulate a contaminatio à la Huber.

            Inputs
            ------
            inv_cdf_s : inverse cumulative distribution function of the sane data
            inv_cdf_o : inverse cumulative distribution function of the contaminated data

            Outputs
            -------
            X (array) : array of dimension N x 2
            index : index of contaminated / sane data (for plot only)
        """

        N = np.max(self.n_sample)
        trial = np.random.binomial(1, 0.5 - self.delta, N) # Sample of length N of bernoulli random variable with proba 0.5+delta, if one, data are contaminated
        X = np.zeros([N,2])
        index = np.where(trial == 0)
        mask = np.ones(N, dtype = bool)
        mask[index] = False # indices of uncontaminated data
        N_ = N - np.sum(mask) # number of uncontaminated data
        self.copula_sane.n_sample = N_
        X[~mask,:] = self.copula_sane.sample(inv_cdf_s)
        N_ = np.sum(mask) # number of contaminated observations
        self.copula_contaminated.n_sample = N_
        X[mask,:] = self.copula_contaminated.sample(inv_cdf_o)
        if show_index == False:
            return X
        else :
            return X, mask
    
    def adversarial_contamination(self, inv_cdf_s, inv_cdf_o, show_index = False):

        N = np.max(self.n_sample)
        nb_contaminated = np.int(N * (0.5-self.delta))
        X = np.zeros([N,2])
        self.copula_sane.n_sample = N
        self.copula_contaminated.n_sample = nb_contaminated

        X = self.copula_sane.sample(inv_cdf_s)
        index = np.argsort(np.linalg.norm(X-np.array([0,0]), axis = 1))
        X = X[index]
        X[0:nb_contaminated] = self.copula_contaminated.sample(inv_cdf_o)
        mask = np.zeros(N, dtype = bool)
        mask[0:nb_contaminated] = True 
        if show_index == False:
            return X
        else :
            return X, mask
    
    def simu(self, inv_cdf):
        """
            Perform Monte Carlo simulation
        """

        output = []

        for k in range(self.n_iter):
            FMado_store = np.zeros(len(self.n_sample))
            obs_all = self.copula.sample(inv_cdf)
            for i in range(0, len(self.n_sample)):
                obs = obs_all[:self.n_sample[i]]
                FMado = self._fmado(obs)
                FMado_store[i] = FMado[0,1]
        
            output_cbind = np.c_[FMado_store, self.n_sample, np.arange(len(self.n_sample))]
            output.append(output_cbind)
        df_FMado = pd.DataFrame(np.concatenate(output))
        df_FMado.columns = ['FMado', "n", "gp"]
        df_FMado['scaled'] = (df_FMado.FMado - df_FMado.groupby('n')['FMado'].transform('mean')) * np.sqrt(df_FMado.n)
        
        return(df_FMado)