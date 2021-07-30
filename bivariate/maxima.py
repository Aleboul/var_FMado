"""
    Produit l'expérience lorsque l'on prend le maximum de variables aléatoires suivant
    une gaussienne bivariée (liée à la Hüsler Reiss)
"""
import numpy as np
import pandas as pd
import monte_carlo
import extreme_value_copula
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
plt.style.use('seaborn-whitegrid')
from matplotlib import cm
n = 100
#n_iter = 1000
#n_sample = [256]
#n_max = [64,128,256]
theta = 0.8
#
#class MaxBivGau(object):
#    """
#        Base class for sample from the maximum of the bivariate gaussian
#
#        Parameters
#        ----------
#            theta (float or None) : Parameter for the copula.
#            n_max ([int]) : Number of data where we took the maximum.
#            n_sample ([int]) : lengths of the sample.
#        Attributes
#        ----------
#        sample (2-D array like, of shape (n_sample,2))
#        
#    """
#
#    def __init__(self, random_seed = None, theta = None, n_sample = None, n_max = None):
#
#        self.random_seed = random_seed
#        self.theta = theta
#        self.n_max = n_max
#        self.n_sample = n_sample
#        self.mean = [5,5]
#        self.cov = [[1, 1 - theta / np.log(self.n_max)], [1 - theta / np.log(self.n_max), 1]]
#
#    def sample(self, inv_cdf = None):
#        sample = np.zeros((self.n_sample,2))
#        for n in range(0, self.n_sample):
#            sample_ = np.random.multivariate_normal(self.mean, self.cov, size = self.n_max)
#            max_ = [np.max(sample_[:,0]), np.max(sample_[:,1])]
#            sample[n] = max_
#
#        return(sample)
#
#x = np.linspace(0.01,0.99,n)
#var_lmbd = pd.DataFrame()
#for n in n_max:
#    value = []
#    var_lmbd_ = pd.DataFrame()
#    copula = MaxBivGau(theta = theta, n_sample= np.max(n_sample), n_max = n)
#    Monte = monte_carlo.Monte_Carlo(n_iter= n_iter, n_sample= n_sample, copula= copula)
#    var = Monte.exec_varlmbd(inv_cdf= None,lmbds= x)
#    var_lmbd_['var_emp'] = var
#    copula = extreme_value_copula.Hussler_Reiss(theta = theta)
#    for lmbd in x:
#        value_ = copula.var_FMado(lmbd)
#        value.append(value_)
#    var_lmbd_['var_theo'] = value
#    var_lmbd_['lmbd'] = x
#    var_lmbd_['n_max'] = n
#    var_lmbd  = var_lmbd.append(var_lmbd_)
#
#print(var_lmbd)
#

with open("output/max_student_M100_n512.txt", "rb") as data:
    b = pickle.load(data)

var_lmbd = pd.DataFrame(b)
x = np.linspace(0.01,0.99,n)
value = []
copula = extreme_value_copula.Student(theta = theta, psi1=3.0)
for lmbd in x:
    value_ = copula.var_FMado(lmbd)
    value.append(value_)

theorical = pd.DataFrame()
theorical['var_theo'] = value
theorical['lambda'] = x
theorical['nmax'] = 512

print(var_lmbd)
print(theorical)
fig, ax = plt.subplots()
sns.scatterplot(data = var_lmbd, x = "lambda", y = "var_emp", hue = "nmax", palette= 'crest', s = 10, alpha = 1.0)
sns.lineplot(data = theorical, x = "lambda", y = "var_theo", hue = "nmax", palette = 'crest',legend = False, lw = 1)
ax.legend(title = r'$\theta$')
ax.set_ylabel(r'$\sigma^2$')
ax.set_xlabel(r'$\lambda$')
plt.savefig("/home/aboulin/Documents/stage/var_FMado/bivariate/output/image_4.pdf")