"""
    Représente les types de contaminations.
    
    Inputs
    ------
    n_sample ([int]) : sample's length
    theta, psi1, psi2 : copula's parameter
    copula_sane : law of the sane observations
    sample_outliers : design of the outliers
"""

from os import spawnlp
import extreme_value_copula
import archimedean
import monte_carlo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
from matplotlib import cm
from scipy.stats import norm
from scipy.stats import expon

def sample_outliers(n):
    x = np.random.uniform(low = 0.95, high = 1.0, size = n)
    y = np.random.uniform(low = 0.0, high = 0.05, size = n)

    sample = np.array([x,y]).T

    return sample

def sample_outliers(n):
    x = np.random.uniform(low = 0.0, high = 0.05, size = n)
    y = np.random.uniform(low = 0.95, high = 1.0, size = n)

    sample = np.array([x,y]).T

    return sample

n_sample = [10000]
theta = 2.5
psi1 = 0.5
psi2 = 1.0
copula_sane = extreme_value_copula.Asy_neg_log(theta = theta, psi1= psi1, psi2= psi2)


Monte = monte_carlo.Monte_Carlo(n_sample= n_sample, copula_sane= copula_sane, sample_outliers = sample_outliers, delta = 0.40)

sample, index = Monte.adversarial_contamination(inv_cdf_s= norm.ppf, show_index= True, unimargin = True)
print(sample, index)

fig, ax = plt.subplots()
ax.plot(sample[index,0], sample[index,1], '.',markersize = 1, alpha = 0.75, color = 'lightcoral')
ax.plot(sample[~index,0], sample[~index,1], '.',markersize = 1, alpha = 0.5, color = 'lightblue')
ax.set_title(r'Adversarial contamination $\delta = 0.4$')
plt.savefig("/home/aboulin/Documents/stage/var_FMado/robust/output/plot_gumbel_adversarial_top_left.pdf")

print(len(sample[index,:]))
