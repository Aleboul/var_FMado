"""
	This code is from Ryan Cotsakis
"""

import numpy as np
import scipy.stats as st

def print_completion(iter_num, num_iters, message=""):
	try:
		milestones = int(num_iters/33)
		if iter_num % milestones == 0:
			print(message + " {}% complete.".format(int(100*iter_num/num_iters)))
	except:
		pass


"""
Generates Brown resnick random fields.

@param X: a vector that represents the x axis. If in two dimensions, the y axis is taken to be the same as the x axis.
@param theta: a tuple of (range, hurst index). Use theta=(range, 1) for the Smith process.
@param dimension: 1 or 2.
"""
class BR_generator():

	gamma = lambda h, theta: np.linalg.norm(h/theta[0])**(2*theta[1]) # this is the semi-variogram of the process

	def __init__(self, X, theta, dimension=1):
		self.d = dimension
		self.X = X
		self.m = len(X)
		self.theta = theta
		self.L = self.frac_brownian_surface_matrix()

	def vectorize_index(self, i):
		if self.d == 2:
			a, b = np.divmod(i, self.m)
			return np.array([self.X[a],self.X[b]])
		else:
			return self.X[i]

	def frac_brownian_surface_matrix(self):
		G = np.zeros((self.m**self.d, self.m**self.d))
		for i in range(self.m**self.d):
			for j in range(self.m**self.d):
				G[i,j] = BR_generator.gamma(self.vectorize_index(i) - self.vectorize_index(j), self.theta)
		U, S, Vh = np.linalg.svd(G)
		L = U @ np.diag(np.sqrt(S)) @ Vh
		return L

	"""
	Returns a multivariate gaussian random variable whose covariance matrix is L@L.T
	"""
	def gaussian_process(self):
		x = st.norm.rvs(size=self.m**self.d)
		return np.copy(self.L@x)

	def simulate_Y(self, i):
		W = self.gaussian_process()
		semivariogram = []
		for j in range(self.m**self.d):
			h = self.vectorize_index(i)-self.vectorize_index(j)
			semivariogram.append(BR_generator.gamma(h,self.theta))
		Y = np.exp(W - W[i] - semivariogram)
		return Y

	def generate(self, verbose=False):
		#print('generating BR')
		Z = np.zeros(self.m**self.d)

		for i in range(self.m**self.d):
			if verbose:
				print_completion(i, self.m**self.d, "Generating BR.")
			zeta = 1/st.expon.rvs()
			while zeta > Z[i]:
				Y = self.simulate_Y(i)
				if i==0 or all(zeta*Y[0:i] < Z[0:i]):
					Z = np.maximum(Z, zeta*Y)
				zeta /= (1 + zeta*st.expon.rvs())
		if self.d == 2:
			return Z.reshape((self.m,self.m))
		else:
			return Z

# brg = BR_generator(np.linspace(0,1,10), (3,1), 2)
# print(brg.generate())
