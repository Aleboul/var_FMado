import math
import numpy as np
import matplotlib.pyplot as plt

def max(a,b):
	if a >= b:
		return a
	else :
		return b

def _ecdf(data):
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

def _dist(X, lmbd):
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
                d = np.linalg.norm((np.power(F_x,1) - np.power(G_y,1)), ord = 1)# - (lmbd) * np.sum((1-np.power(F_x,lmbd))) - (1-lmbd) * np.sum((1 - np.power(G_y, 1 - lmbd))) # formula of the normalized lmbd madogram estimator, see Naveau 2009
                dist[i,j] = d
                dist[j,i] = d
    return dist

def _fmado(X, lmbd):
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
        Femp = _ecdf(X_vec)
        V[:,p] = Femp
    Fmado = _dist(np.transpose(V), lmbd) / (2 * Tnb) #+ (1/2) * ((1 - lmbd*(1-lmbd))/ ((2-lmbd)*(1+lmbd)))
    return Fmado

def rsmith1d(coord, center, edge, nObs, nSites, var):
	"""
		This function generates random fields for the 1d smith model
		
		Inputs
		------
		 coord : the coordinates of the locations
		center : the center of the compact set - here I use an interval
		  edge : the length of the interval
		  nObs : the number of observations to be generated
		nSites : the number of locations
		   var : the variance of the univariate normal density
		Outputs
		-------
		   ans : the generated random field
	"""
	ans = np.zeros(nSites * nObs)
	uBound = math.sqrt(1/(2*math.pi*var)) 
	if(var <=0):
		raise ValueError('The variance should be strictly positive! \n')
		
	"""
		We first center the coordinates to avoid repetition of
		unnecessary operations in the wile loop
	"""
	for i in range(0,nSites):
		coord[i] -= center
		
	"""
		Simulation according to the Schlater methodology. The compact
		set need to be inflated first
	"""
	edge = 6.92 * math.sqrt(var)
	lebesgue = edge
	
	for i in range(0,nObs):
		poisson = 0.0
		nKO = nSites
		
		while nKO > 0:
			"""
				The stopping rule is reached when nKO = 0 i.e. when each site
				satisfies the condition in Eq. (8) of Schlather (2002)
			"""
			poisson += np.random.exponential(size = 1)
			ipoisson = 1 / poisson; thresh = uBound * ipoisson
			
			# We simulate points uniformly in [-r/2, r/2]
			u = edge * np.random.uniform(-0.5,0.5,1)
			
			#nKo = nSites
			for j in range(0, nSites):
				# This is the normal density with 0 mean and variance var
				y = math.exp(-(coord[j] - u) * (coord[j] - u) / (2*var)) * thresh
				ans[i + j * nObs] = max(y, ans[i+j*nObs])
				nKO -=int(thresh <= ans[i+j*nObs])
	"""
		Lastly, we multiply by the lebesgue measure of the dilated
		compact set
	"""
	for i in range(0, nSites):
		ans[i] *= lebesgue
		
	ans = ans.reshape(nObs, nSites)
	return ans

def _mse(h,lmbd, nobs, niter):
    """
        Return the MSE of the lambda-FMadogram where data are simulated from a
        max-stable smith process.

        Inputs
        ------
        h : distance of the two locations
        lmbd : lmbd use for the Madogram
        nobs : length of the sample
        niter : number of simulation

    """

coord = np.array([0,0.01])
center = np.mean(coord)
edge = np.abs(coord[-1] - coord[0])
nObs = 100
nSites = coord.shape[0]
var = edge / 5

ans = rsmith1d(coord, center, edge, nObs, nSites,var)

print(ans)

#FMado = _fmado(ans, 0.5)
#print(FMado[0,1])

