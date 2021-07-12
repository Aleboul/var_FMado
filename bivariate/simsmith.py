import math
import numpy as np
import matplotlib.pyplot as plt

def max(a,b):
	if a >= b:
		return a
	else :
		return b

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
	
def rsmith2d(coord, center, edge, nObs, nSites, grid, cov11, cov12, cov22):
	"""
		Thid function generates random fields for the 2d smith model
		
		Inputs
		------
		coord : the coordinates of the locations
		center : the center of the compact set - here I use a square
		edge : the length of the edge of the square
		nObs : the number of observations to be generated
		grid : Does coord specifies a grid?
		nSites : the number of locations
		covXX : the parameters of the bivariate normal density
		
		Output
		------
		ans : the generated random field
	"""
	ans = np.zeros(nObs * nSites * nSites)
	det = cov11 * cov22 - cov12 * cov12
	uBound = math.sqrt(1/(2*math.pi*det)) ; itwiceDet = 1 / (2*det)
	
	if ((det <= 0) | (cov11 <=0)):
		raise ValueError("The covariance matrix isn't semi-definite positive! \n")
		
	"""
		We first center the coordinates to avoid repetition of
		unnecessary operations in the wile loop
	"""
	for i in range(0, nSites):
		coord[i,0] -= center[0]
		coord[i,1] -= center[1]
		
	"""
		Simulation according to Schlater methodology. The compact
		set need to be inflated first
	"""
	edge += 6.92*math.sqrt(max(cov11, cov22))
	lebesgue = edge * edge
	if grid == True:
		## simulation part if a grid is used
		for i in range(1,nObs):
			poisson = 0.0
			nKO = nSites * nSites
			
			while nKO > 0:
				"""
					The stopping rule is reached when nKO = 0 i.e. when each site
					satisfies the condition in Eq. (8) of Schlater (2002)
				"""
				poisson += np.random.exponential(size = 1)
				ipoisson = 1 / poisson ; thresh = uBound * ipoisson
				
				# We simulate points uniformly in [-r/2,r/2]^2
				u1 = edge * np.random.uniform(-0.5,0.5, 1)
				u2 = edge * np.random.uniform(-0.5,0.5, 1)
				
				nKO = nSites * nSites
				for j in range(0,nSites):
					for k in range(0,nSites):
						"""
							This is the bivariate normal density with 0 mean and
							cov. matric [cov11, cov12 ; cov12, cov22]
						"""
						y = math.exp((-cov22*(coord[j,0] - u1) * (coord[j,0] - u1) + 2 * cov12 * 
									 (coord[j,0] - u1) * (coord[k,1] - u2) - cov11 *
									 (coord[k,1] - u2) * (coord[k,1] - u2)) *
									 itwiceDet) * thresh
						ans[j + k * nSites + i * nSites * nSites] = max(y, ans[j + k * nSites + i * nSites * nSites])
						
						nKO -= int(thresh <= ans[j + k * nSites + i * nSites * nSites])
						
	else:
		# Simulation parf if a grid isn't used
		for i in range(0, nObs):
			poisson = 0
			nKO = nSites
			
			while nKO > 0 :
				"""
					The stopping rule is reached when nKO = 0 i.e. when each site
					satisfies the condition in Eq. (8) of Schlater (2002)
				"""
				poisson += np.random.exponential(size = 1)
				ipoisson = 1 / poisson ; thresh = uBound * ipoisson
				
				# We simulate points uniformly in [-r/2, r/2]^2
				u1 = edge * np.random.uniform(-0.5,0.5,1)
				u2 = edge * np.random.uniform(-0.5,0.5,1)
				
				nKO = nSites
				for j in range(0, nSites):
					"""
						This is the bivariate normal density with 0 mean and
						cov. matrix [cov11, cov12; cov12, cov22]
					"""
					y = math.exp((-cov22*(coord[j,0] - u1) * (coord[j,0] - u1) + 2 * cov12 * 
									 (coord[j,0] - u1) * (coord[k,1] - u2) - cov11 *
									 (coord[k,1] - u2) * (coord[k,1] - u2)) *
									 itwiceDet) * thresh
					
					ans[i+j * nObs] = max(y, ans[i+j * nObs])
					
					nKO -= int(thresh <= ans[i + j * nObs])
	
	"""
		Lastly, we multiply by the Lebesgue measure of the dilated
		compact set
	"""
	print(ans)
	if grid :
		for i in range(0, nSites * nSites * nObs):
			ans[i] *= lebesgue[0]
			
	else:
		for i in range(0, nSites * nObs):
			ans[i] *= lebesgue
			
	return ans
coord = np.linspace(0,10,100)
center = np.mean(coord)
edge = np.abs(coord[-1] - coord[0])
nObs = 1
nSites = coord.shape[0]
var = 16.0

ans = rsmith1d(coord, center, edge, nObs, nSites,var)

print(ans)

#x = np.linspace(0,10,10)
#y = np.linspace(0,10,10)
#coord = np.vstack((x,y)).T
#nSites = coord.shape[0]
#center = coord.mean(axis=0)
#edge = np.array([coord[-1,0] - coord[0,0], coord[-1,1] - coord[0,1]])
#
#ans = rsmith2d(coord, center, edge, 1, nSites, cov11 = 9/8, cov12 = 0, cov22=9/8, grid = True)