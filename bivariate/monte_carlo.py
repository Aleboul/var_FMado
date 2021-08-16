import numpy as np
import pandas as pd
from tqdm import tqdm

class Monte_Carlo(object):
    """
        Base class for Monte-Carlo simulations
        Inputs
        ------
            n_iter (int): number of Monte Carlo simulation
            n_sample (list of int or [int]): multiple length of sample
            lmb (float): value of lmbd
            random_seed (Union[int, None]) : seed for the random generator
        Attributes
        ----------
            n_sample (list[int]) : several lengths used for estimation
            lmbd (float) : parameter for the lambda-FMadogram
            lmbd_interval ([lower, upper]) : interval of valid thetas for the given lambda-FMadogram
            simu (DataFrame) : results of the Monte-Carlo simulation
                - FMado : the value of estimator
                - n : length of the sample
                - gp : number of iteration
                - scaled : ::math:: \sqrt{T}(\hat{\nu}_T (\lambda) - \nu(\lambda))
    """

    n_iter = None
    n_sample = []
    lmbd = None
    lmbd_interval = []
    random_seed = None
    copula = None

    def __init__(self, n_iter = None, n_sample = [], lmbd = 0.5, random_seed = None, copula = None):
        """
            Initialize Monte_Carlo object
        """
        self.n_iter = n_iter
        self.n_sample = n_sample
        self.lmbd = lmbd
        self.copula = copula

    def check_lmbd(self):
        """
            Validate the lmbd
            This method is used to assert the lmbd insert by the user.
            Raises :
                ValueError : If lmbd is not in :attr:`lmbd_interval`
        """

        lower, upper = self.lmbd_interval
        if (not lower <= sef.lmbd <= upper):
            message = "The lmbd value {} is out of limits for the given estimation."
            raise ValueError(message.format(self.lmbd))

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


    def _dist(self, X):
        """
	    	Compute matrix of F-Madogram (a une constante pres) using the empirical cumulative distribution function
    
	    	Inputs
	    	------
	    	X : a matrix composed of ecdf
	    	lmbd : a parameter between 0 and 1
	    	Outputs
	    	-------
	    	A matrix with quantity equals to 0 if i=j (diagonal) and equals to sum_t=1^T |F(X_t)^{\lmbd} - G(Y_t)^{1-\lmbd}| if i \neq j
	    """

        ncols = X.shape[1]
        nrows = X.shape[0]
        dist = np.zeros([nrows, nrows]) # initialization
        for i in range(0, nrows):
            for j in range(0,i):
                if i==j :
                    dist[i,i] = 0
                else :
                    F_x = np.squeeze(X[j,:])
                    G_y = np.squeeze(X[i,:])
                    d = np.linalg.norm((np.power(F_x,self.lmbd) - np.power(G_y,1-self.lmbd)), ord = 1) - (self.lmbd) * np.sum((1-np.power(F_x,self.lmbd))) - (1-self.lmbd) * np.sum((1 - np.power(G_y, 1 - self.lmbd))) # formula of the normalized lmbd madogram estimator, see Naveau 2009
                    dist[i,j] = d
                    dist[j,i] = d
        return dist

    def _fmado(self, X):
        """
		    This function computes the lmbd FMadogram
    
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
        Fmado = self._dist(np.transpose(V)) / (2 * Tnb) + (1/2) * ((1 - self.lmbd*(1-self.lmbd))/ ((2-self.lmbd)*(1+self.lmbd)))

        return Fmado
    
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

    def exec_varlmbd(self, inv_cdf, lmbds, n_lmbds = 50):
        if lmbds is None:
            lmbds = np.linspace(0,1,n_lmbds)
        else :
            lmbds = lmbds
        for i,n in enumerate(self.n_sample):
            var_lmbd = []
            for lmbd in tqdm(lmbds):
                self.lmbd = lmbd
                output = self.simu(invècdf)
                output = output['scaled'].var()
                var_lmbd.append(output)

        return var_lmbd
