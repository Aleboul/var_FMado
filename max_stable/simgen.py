import numpy as np
import br_generator
import pickle
import math
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.stats import norm

"""
	Extract a sample from smith process generated with R and work with it (plot several figures).
"""

def V(h, lmbd):
    """
        Compute V_h(\lmbd, 1-lmbd)
    """
    V_ = (1/lmbd) * norm.cdf(h/(2*sigma) + (sigma / h) * math.log((1-lmbd)/lmbd)) + 1 / (1-lmbd) * norm.cdf(h/(2*sigma) + (sigma / h) * math.log(lmbd/(1-lmbd)))
    return V_

def true_FMado(h, lmbd):
    """
        Compute the real value of a madogram for Smith's process
    """
    if (lmbd == 0) | (lmbd == 1) :
        value_ = 1/4
    else: 
        c_ = 3/(2*(1 + lmbd) * (1 + 1 - lmbd))
        value_ = V(h, lmbd) / (1+V(h,lmbd)) - c_
    return(value_)

def matrix_FMado(h, lmbd):
    """
        Compute a matrix of lambda-FMadogram for Smith's process

        Inputs
        ------
           h : an array of distance
        lmbd : an array of lambda

        Outputs
        -------
        np.array
    """
    _horizon = np.delete(h,0)
    ans = np.zeros((len(_horizon),len(lmbd)))
    for i,_h in enumerate(_horizon):
        for j,_lmbd in enumerate(lmbd):  
            ans[i,j] = true_FMado(_h, _lmbd)

    return(ans)

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
                d = np.linalg.norm((np.power(F_x,lmbd) - np.power(G_y,1-lmbd)), ord = 1) - (lmbd) * np.sum((1-np.power(F_x,lmbd))) - (1-lmbd) * np.sum((1 - np.power(G_y, 1 - lmbd))) # formula of the normalized lmbd madogram estimator, see Naveau 2009
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
    Fmado = _dist(np.transpose(V), lmbd) / (2 * Tnb) + (1/2) * ((1 - lmbd*(1-lmbd))/ ((2-lmbd)*(1+lmbd)))
    return Fmado

def simgen(h, lmbd, niter, nsample, sigma):
    """
        Compute a matrix of lambda-FMadogram estimate 
        from a max stable smith random process
        varying with h and theta

        Inputs
        ------
              h : array of distance
           lmbd : array of lambda
          niter : number of Monte Carlo simulation
        nsample : length of sample  
    """
    ans = [] # output
    #ans.append(h)
    #ans.append(lmbd)
    brg = br_generator.BR_generator(h, (np.sqrt(2)*sigma,1),1)
    _horizon = np.delete(h,0)
    print(_horizon)
    for n in tqdm(range(0, niter)):
        _sample = np.zeros((nsample,len(h))) # generate sample of the Smith process
        _lmbd_FMado = np.zeros((len(_horizon),len(lmbd)))
        for i in range(0,nsample):
            _sample[i] = brg.generate()
        for i,_h in enumerate(_horizon):
            print(i)
            for j,_lmbd in enumerate(lmbd): 
                _lmbd_FMado[i,j] = _fmado(_sample[:,[0,i+1]], _lmbd)[0,1]

        ans.append(_lmbd_FMado)
    return(ans)

def Matrix_MSE(list_matrices, h, lmbd):
    """
        Return MSE matrice where MSE is computed component wise
        with a list of matrices

        Inputs
        ------
        list_matrices : a list of matrices given by simgen
                    h : array of h used by simgen
                 lmbd : array of lambda used by simgen

        Outputs
        -------
        array of component wise MSE of \nu(h, lambda)
    """
    V_ = matrix_FMado(h, lmbd)
    means = np.array([sum(x) / niter for x in zip(*list_matrices)])
    biais = np.power(means - V_,2)
    X_2 = np.array([sum(np.power(x,2)) for x in zip(*list_matrices)])
    variance =np.array(X_2 / niter - np.power(means,2))
    MSE = np.power(biais + variance,1)
    return(MSE)

def plot3d(Matrix,h, lmbd, filename,zlabel = r'$MSE$',save = False, Matrix_2 = None):
    """
        Plot a matrix

        Inputs
        ------
            Matrix : matrix depending on h and lmbd
                 h : array on h used by simgen
              lmbd : array on lambda used by simgen

    """
    _horizon = np.delete(h,0)
    _horizon, lmbd = np.meshgrid(_horizon, lmbd)
    print(Matrix.shape)
    print(_horizon.shape)
    print(lmbd.shape)
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(_horizon, lmbd, Matrix, alpha = 0.5, color = 'darkblue')
    if Matrix_2 is not None:
        ax.plot_surface(_horizon, lmbd, Matrix_2, alpha = 0.5, color = 'salmon')
    #ax.plot_wireframe(_horizon, lmbd, Matrix, rstride = 10, cstride=10)
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$\lambda$')
    ax.set_zlabel(zlabel)
    if save :
        path = os.path.join('/home/aboulin/Documents/stage/var_FMado/max_stable/output/', filename)
        plt.savefig(path)
    else :
        return fig, ax

def exec():
    means = np.array([sum(x) / niter for x in zip(*b)])
    V_ = matrix_FMado(h, lmbd)
    X_2 = np.array([sum(np.power(x,2)) for x in zip(*b)])
    variance = nsample*np.array(X_2 / niter - np.power(means,2))
    mse = Matrix_MSE(b, h, lmbd)
    plot3d(means.T, h, lmbd, filename= 'lMadogram_smith_300_1024.pdf', zlabel = r'$\nu$',save = True, Matrix_2= V_.T)
    plot3d(variance.T, h, lmbd, zlabel= r'$\sigma^2$', filename= 'variance_smith_300_1024.pdf', save = True)
    plot3d(mse.T, h, lmbd, filename = 'MSE_smith_300_1024.pdf', save = True)

sigma = np.sqrt(5)
h = np.linspace(0,20,21)
lmbd = np.linspace(0.0,1,11)
niter = 300
nsample = 1024

with open("data_1024.txt", "rb") as data:
    b = pickle.load(data)

exec()
