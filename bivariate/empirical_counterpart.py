"""
    Ce code produit des simulations de Monte Carlo pour un model sélectionné.
    Il réalise pour une grille lambda choisi une contrepartie empirique et 
    ajoute sa valeur théorique dessus.

    Inputs
    ------
    n (int): grid's length of the lambda-axis
    n_iter (int): number of samples
    n_sample ([int]) : length of samples
    theta ([float]), psi1 (floatt), psi2 (float) : parameters of the copula
"""

import extreme_value_copula
import monte_carlo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
from matplotlib import cm
from scipy.stats import norm


n = 1000
n_iter = 500
n_sample = [256]
theta = [1.0]
psi1 = 1.0
psi2 = 1.0
random_seed = 42

x = np.linspace(0.01,0.99,n)
var_lmbd = pd.DataFrame()
for theta_ in theta:
    value = []
    var_lmbd_ = pd.DataFrame()
    copula = extreme_value_copula.Asy_log(random_seed = 42, theta = theta_, n_sample = np.max(n_sample), psi1= psi1, psi2= psi2)
    Monte = monte_carlo.Monte_Carlo(n_iter= n_iter, n_sample= n_sample, random_seed= random_seed, copula= copula)
    var = Monte.exec_varlmbd(lmbds= x, inv_cdf = norm.ppf)
    var_lmbd_['var_emp'] = var
    for lmbd in x:
        value_ = copula.var_FMado(lmbd)
        value.append(value_)
    var_lmbd_['var_theo'] = value
    var_lmbd_['lmbd'] = x
    var_lmbd_['theta'] = theta_
    var_lmbd  = var_lmbd.append(var_lmbd_)

print(var_lmbd)

fig, ax = plt.subplots()
sns.scatterplot(data = var_lmbd, x = "lmbd", y = "var_emp", hue = "theta", palette= 'crest', s = 10, alpha = 0.5)
sns.lineplot(data = var_lmbd, x = "lmbd", y = "var_theo", hue = "theta", palette = 'crest',legend = False, lw = 1)
ax.legend(title = r'$\theta$')
ax.set_ylabel(r'$\sigma^2$')
ax.set_xlabel(r'$\lambda$')
plt.savefig("/home/aboulin/Documents/stage/var_FMado/bivariate/output/test.pdf")