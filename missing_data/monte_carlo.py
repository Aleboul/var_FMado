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
	
	def __init__(self, n_iter = None, n_sample = [], lmbd = 0.5, random_seed = None, copula = None, p = [1.0,1.0]):
		"""
			Initialize Monte_Carlo object
		"""
		
		self.n_iter = n_iter
		self.n_sample = n_sample
		self.lmbd = lmbd
		self.copula = copula
		self.p = p
		
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
		
	def _dist(self, X, miss, corr):
		"""	
			Compute matrix of F-Madogram using the empirical cumulative distribution function
			
			Inputs
			------
			   X : a matrix composed of uniform ecdf
			lmbd : a parameter between 0 and 1
			miss : array indicates missing
			corr : if true, return corrected version
			
			Outputs
			-------
			A matrix with quantity equals to 0 if i=j (diagonal) and equals to sum_t=1^T |F(X_t)^{\lmbd} - G(Y_t)^{1-\lmbd}| if i \neq j
		"""
		
		ncols = X.shape[1]
		nrows = X.shape[0]
		F_x = np.squeeze(X[0,:])
		G_y = np.squeeze(X[1,:])
		if corr :
			dist = np.linalg.norm((np.power(F_x,self.lmbd) - np.power(G_y,1-self.lmbd)) * (miss[0] * miss[1]), ord = 1) - self.lmbd * np.sum((1-np.power(F_x,self.lmbd))* (miss[0] * miss[1])) - (1-self.lmbd) * np.sum((1-np.power(G_y,1-self.lmbd))* (miss[0] * miss[1]))
		else :
			dist = np.linalg.norm((np.power(F_x,self.lmbd) - np.power(G_y,1-self.lmbd)) * (miss[0] * miss[1]), ord = 1)
		return dist
		
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
		if corr :
			FMado = self._dist(np.transpose(V), miss, corr) / (2 * np.sum(miss[0] * miss[1])) + 0.5 * (1-self.lmbd + np.power(self.lmbd,2)) / ((1+self.lmbd) / (1+1-self.lmbd))
		else :
			FMado = self._dist(np.transpose(V), miss, corr) / (2 * np.sum(miss[0] * miss[1]))

		return FMado
		
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
			miss_all = [np.random.binomial(1,self.p[0],np.max(self.n_sample)),np.random.binomial(1,self.p[1],self.n_sample)]
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