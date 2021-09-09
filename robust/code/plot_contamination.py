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

#def inv_cdf_o(u):
#    return norm.ppf(u, 2, np.sqrt(0.5))
#
#def sample_outliers(n):
#    copula_contaminated = extreme_value_copula.Asy_log(n_sample = n,random_seed = 42, theta = 1.0, psi1 = 1.0, psi2 = 1.0)
#    return copula_contaminated.sample(inv_cdf= inv_cdf_o)

def sample_outliers(n):
    x = np.random.uniform(low = 2.0, high = 3.0, size = n)
    y = np.random.uniform(low = -3.0, high = -2.0, size = n)

    sample = np.array([x,y]).T

    return sample

#def sample_outliers(n):
#    x = np.random.uniform(low = -3.0, high = -2.0, size = n)
#    y = np.random.uniform(low = 2.0, high = 3.0, size = n)
#
#    sample = np.array([x,y]).T
#
#    return sample

#def sample_outliers(n):
#    length = np.random.uniform(3.5,4.0,n)
#    angle = np.pi * np.random.uniform(0,2,n)
#
#    x = length * np.cos(angle)
#    y = length * np.sin(angle)
#
#    sample = np.array([x,y]).T
#
#    return sample

n_sample = [10000]
theta = 10
psi1 = 0.5
psi2 = 1.0
copula_sane = extreme_value_copula.Asy_neg_log(random_seed = 42, theta = theta, psi1= psi1, psi2= psi2)


Monte = monte_carlo.Monte_Carlo(n_sample= n_sample, copula_sane= copula_sane, sample_outliers = sample_outliers, delta = 0.40)

sample, index = Monte.adversarial_contamination(inv_cdf_s= norm.ppf, show_index= True)

print(sample, index)

fig, ax = plt.subplots()
ax.plot(sample[index,0], sample[index,1], '.',markersize = 1, alpha = 0.75, color = 'lightcoral')
ax.plot(sample[~index,0], sample[~index,1], '.',markersize = 1, alpha = 0.5, color = 'lightblue')
ax.set_title(r'Adversarial contamination $\delta = 0.4$')
plt.savefig("/home/aboulin/Documents/stage/var_FMado/robust/output/test.pdf")

print(len(sample[index,:]))
