"""
    Représente les types de contaminations.

    Comparaison entre le FMadogramme et le MoN-FMadogramme
    
    Inputs
    ------

    n_sample : sample's length
    n_iter : number of estimator taken to compute the biais
    theta, psi1, psi2 = copula's parameters
    copula_sane : sane data's law
    sample_outliers : design of outliers

"""

import var_FMado.reproducevarFMado.missingdata.src.archimedean
import var_FMado.reproducevarFMado.missingdata.src.extreme_value_copula
import var_FMado.reproducevarFMado.missingdata.src.monte_carlo
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
theta = 1.5
psi1 = 1.0
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
