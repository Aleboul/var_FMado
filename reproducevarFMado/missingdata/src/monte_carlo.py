import numpy as np
import pandas as pd
from tqdm import tqdm

class Monte_Carlo(object):
	"""
		Base class for Monte-Carlo simulations
		Inputs
		------
			                   n_iter (int) : number of Monte Carlo simulation
			n_sample (list of int or [int]) : multiple length of sample
			                   lmbd (float) : value of lambda	
			 random_seed (Union[int, None]) : seed for the random generator
						p (list of [float]) : array of probabilities of presence
							copula (object) : law of the vector of the uniform margin
								copula_miss : dependence modeling of the couple (I,J)
		Attributes
		----------
			          n_sample (list[int]) : several lengths used for estimation
							  lmbd (float) : parameter for the lambda-FMadogram
			lmbd_interval ([lower, upper]) : interval of valid thetahs for the given lambda-FMadogram
						  simu (DataFrame) : results of the Monte-Carlo simulation
							 - FMado : the value of estimator
							     - n : length of the sample
							    - gp : number of iteration
							- scaled : ::math:: \sqrt{T}(\hat{\nu}_T (\lambda) - \nu(\lambda))
	"""
	
	n_iter = None
	n_sample = []
	lmbd = None
	lmbd_interval = None
	random_seed = None
	copula = None
	copula_miss = None
	
	def __init__(self, n_iter = None, n_sample = [], lmbd = 0.5, random_seed = None, copula = None, p = [1.0,1.0], copula_miss = None):
		"""
			Initialize Monte_Carlo object
		"""
		
		self.n_iter = n_iter
		self.n_sample = n_sample
		self.lmbd = lmbd
		self.copula = copula
		self.p = p
		self.copula_miss = copula_miss
		
	def check_lmbd(self):
		"""
			Validate the lmbd
			This method is used to assert the lambda inserted by the user
			Raises :
				ValueError : If lambda is not in :attr:`lmbd_interval`
		"""
		lower, upper = self.lmbd_interval
		if (not lower <= self.lmbd <= upper):
			message = "The lmbd value {} is out of limits for the given estimation."
			raise ValueError(message.format(self.lmbd))
			
	def _ecdf(self, data, miss):
		"""
			Compute ECDF of the data
			Inputs
			------
			data : array of data
			Outputs
			-------
			Empirical cumulative distribution function of a uniform law.
		"""
		
		index = np.argsort(data)
		ecdf = np.zeros(len(index))
		for i in index:
			ecdf[i] = (1.0 / np.sum(miss)) * np.sum((data <= data[i]) * miss)
		return ecdf
		
	def _fmado(self, X, miss, corr) :
		"""
			This function computes the lambda-FMadogram
			
			Inputs
			------
			   X : a matrix
			lmbd : constant between 0 and 1
			miss : list of array of indicator
			corr : If true, return corrected version of lambda-FMadogramme
			Outputs
			-------
			matrix equals to 0 if i = j and equals to |F(X)^{\lmbd} - G(Y)^{1-\lmbd}|
		"""
		
		Nnb = X.shape[1]
		Tnb = X.shape[0]
		
		V = np.zeros([Tnb, Nnb])
		for p in range(0, Nnb):
			X_vec = np.array(X[:,p])
			Femp = self._ecdf(X_vec, miss[p])
			V[:,p] = Femp
		if corr == True:

			FMado = (np.linalg.norm((np.power(V[:,0], self.lmbd) - np.power(V[:,1], 1-self.lmbd))* (miss[0] * miss[1]), ord = 1) - (self.lmbd) * np.sum((1-np.power(V[:,0],self.lmbd))* (miss[0] * miss[1])) - (1-self.lmbd) * np.sum((1-np.power(V[:,1],1-self.lmbd))* (miss[0] * miss[1]))) / (2 * np.sum(miss[0] * miss[1])) + 0.5 * (1-self.lmbd + np.power(self.lmbd,2)) / ((1+self.lmbd) / (1+1-self.lmbd))
		else :
			FMado = np.linalg.norm((np.power(V[:,0],self.lmbd) - np.power(V[:,1], 1-self.lmbd))* (miss[0] * miss[1]), ord = 1) / (2 * np.sum(miss[0] * miss[1]))

		return FMado

	def _gen_missing(self):
		"""
			This function returns an array max(n_sample) \times 2 of binary indicate missing of X or Y.
			Dependence between (I,J) is given by copula_miss. The idea is the following
			I \sim Ber(p[0]), J \sim Ber(p[1]) and (I,J) \sim Ber(copula_miss(p[0], p[1])).
			
			We simulate it by generating a sample (U,V) of length max(n_sample) from copula_miss.

			Then, X = 1 if U \leq p[0] and Y = 1 if V \leq p[1]. These random variables are indeed Bernoulli.
			
			Also \mathbb{P}(X = 1, Y = 1) = \mathbb{P}(U\leq p[0], V \leq p[1]) = C(p[0], p[1])
		"""
		if self.copula_miss is None:
			return np.array([np.random.binomial(1,self.p[0],np.max(self.n_sample)),np.random.binomial(1,self.p[1],self.n_sample)])
		else :
			sample_ = self.copula_miss.sample_unimargin()
			miss_ = np.array([1 * (sample_[:,0] <= self.p[0]), 1*(sample_[:,1] <= self.p[1])])
			return miss_
		
	def simu(self, inv_cdf, corr = {False, True, "Both"}):
		"""
			Perform Monte Carlo simulation
		"""
		
		output = []
		
		for k in range(self.n_iter):
			FMado_store = np.zeros(len(self.n_sample))
			if corr == "Both":
				FMado_corr_store = np.zeros(len(self.n_sample))
			obs_all = self.copula.sample(inv_cdf)
			miss_all = self._gen_missing()
			for i in range(0,len(self.n_sample)):
				obs = obs_all[:self.n_sample[i]]
				miss = [miss_all[0][:self.n_sample[i]],miss_all[1][:self.n_sample[i]]]
				if corr == "Both":
					FMado = [self._fmado(obs, miss, corr = True), self._fmado(obs, miss, corr = False)]
					FMado_store[i], FMado_corr_store[i] = FMado[0], FMado[1]
				else :
					FMado = self._fmado(obs, miss, corr)
					FMado_store[i] = FMado

			if corr == "Both":
				output_cbind = np.c_[FMado_store, FMado_corr_store, self.n_sample, np.arange(len(self.n_sample))]
			else:
				output_cbind = np.c_[FMado_store, self.n_sample, np.arange(len(self.n_sample))]
			output.append(output_cbind)
		df_FMado = pd.DataFrame(np.concatenate(output))
		if corr == "Both":
			df_FMado.columns = ['FMado', 'FMado_corr', 'n', 'gp']
			df_FMado['scaled'] = (df_FMado.FMado - self.copula.true_FMado(self.lmbd)) * np.sqrt(df_FMado.n)
			df_FMado['scaled_corr'] = (df_FMado.FMado_corr - self.copula.true_FMado(self.lmbd)) * np.sqrt(df_FMado.n)
		else:
			df_FMado.columns = ['FMado', 'n', 'gp']
			df_FMado['scaled'] = (df_FMado.FMado - self.copula.true_FMado(self.lmbd)) * np.sqrt(df_FMado.n)
		
		return(df_FMado)
		
		
	def exec_varlmbd(self, lmbds, inv_cdf, n_lmbds = 50, corr = True):
		"""
			Performs Monte Carlo simulation in order to compute the empirical variance for
			a given grid of lambda

			Inputs
			------
			lmbds : array given the grid of lambda
			n_lmbds : if a grid is not given, split the interval [0,1] in a chain with n_lmbds number
			corr : version of the estimator.

			Return
			------
			array of variance for a given lambda.
		"""
		if lmbds is None :
			lmbds = np.linspace(0.01,0.99, n_lmbds)
		else :
			lmbds = lmbds
		for i, n in enumerate(self.n_sample):
			var_lmbd = []
			for lmbd in tqdm(lmbds):
				self.lmbd = lmbd
				if corr == "Both":
					output = self.simu(inv_cdf, corr)
					output = [output['scaled'].var(), output['scaled_corr'].var()]
				else :
					output = self.simu(inv_cdf)
					output = output['scaled'].var()

				var_lmbd.append(output)
				
		return var_lmbd