import numpy as np
import pandas as pd
from tqdm import tqdm

class Monte_Carlo(object):
    """
        Base class for Monte-Carlo simulations
        Inputs
        ------
            copula_sane : law of the bivariate sane random vector
            sample_outliers : law of the bivariate outlier random vector
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

    def __init__(self, n_iter = None, n_sample = [], lmbd = 0.5, random_seed = None, copula_sane = None, sample_outliers = None, K = None, delta = None):
        """
            Initialize Monte_Carlo object
        """
        self.n_iter = n_iter
        self.n_sample = n_sample
        self.lmbd = lmbd
        self.copula_sane = copula_sane
        self.sample_outliers = sample_outliers
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
        Fmado = np.linalg.norm(V[:,0] - V[:,1], ord = 1) / (2 * Tnb)

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
        sample_ = X
        np.random.shuffle(sample_)
        n = len(sample_)
        if n % self.K != 0:
            N = (n//self.K) * self.K
            sample_ = sample_[0:N]
        sample_ = np.split(sample_, self.K)
        F_MoN = np.median([self._fmado(sample_[k]) for k in range(0,self.K)], axis = 0)
        return F_MoN
        
    def huber_contamination(self, inv_cdf_s, show_index = False):
        """
            Simulate a contamination à la Huber.

            Inputs
            ------
            inv_cdf_s : inverse cumulative distribution function of the sane data

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
        #self.copula_contaminated.n_sample = N_
        X[mask,:] = self.sample_outliers(N_)
        if show_index == False:
            return X
        else :
            return X, mask
    
    def adversarial_contamination(self, inv_cdf_s, show_index = False):
        """
            Simulate adversarial contamination, the criteria chosen is data are close to zero
            
            Inputs
            ------
            inv_cdf_s : inverse cumulative distribution function of the sane data

            Outputs
            -------
            X (array) : array of dimension N x 2
            index : index of contaminated / sane data (for plot only)
        """

        N = np.max(self.n_sample)
        nb_contaminated = np.int(N * (0.5-self.delta))
        X = np.zeros([N,2])
        self.copula_sane.n_sample = N
        #self.copula_contaminated.n_sample = nb_contaminated

        X = self.copula_sane.sample(inv_cdf_s)
        index = np.argsort(np.linalg.norm(X-np.array([0,0]), axis = 1))
        X = X[index]
        X[0:nb_contaminated] = self.sample_outliers(nb_contaminated)
        mask = np.zeros(N, dtype = bool)
        mask[0:nb_contaminated] = True 
        if show_index == False:
            return X
        else :
            return X, mask
    
    def simu(self, inv_cdf_s, contamination = {"Huber", "Adversarial"}):
        """
            Perform Monte Carlo simulation
        """

        output = []

        for k in tqdm(range(self.n_iter)):
            FMado_store = np.zeros(len(self.n_sample))
            FMado_MoN_store = np.zeros(len(self.n_sample))
            if contamination == "Huber" :
                obs_all = self.huber_contamination(inv_cdf_s= inv_cdf_s)
            if contamination == "Adversarial":
                obs_all = self.adversarial_contamination(inv_cdf_s= inv_cdf_s)
            for i in range(0, len(self.n_sample)):
                obs = obs_all[:self.n_sample[i]]
                FMado_store[i] = self._fmado(obs)
                FMado_MoN_store[i] = self._fmado_MoN(obs)

            output_cbind = np.c_[FMado_store, FMado_MoN_store, self.n_sample, np.arange(len(self.n_sample))]
            output.append(output_cbind)

        df_FMado = pd.DataFrame(np.concatenate(output))
        df_FMado.columns = ['FMado', 'FMado_MoN', "n", "gp"]
        df_FMado['scaled'] = (df_FMado.FMado - self.copula_sane.true_FMado())
        df_FMado['scaled_MoN'] = (df_FMado.FMado_MoN - self.copula_sane.true_FMado())
        
        return(df_FMado)