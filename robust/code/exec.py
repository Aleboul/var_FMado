import extreme_value_copula
import monte_carlo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
from matplotlib import cm
from scipy.stats import norm
from scipy.stats import expon

n_sample = [10000]
theta = 2.0
psi1 = 1.0
psi2 = 1.0
copula_sane = extreme_value_copula.Asy_log(random_seed = 42, theta = theta, psi1= psi1, psi2= psi2)
copula_contaminated = extreme_value_copula.Asy_neg_log(random_seed = 42, theta = 1.0, psi1 = 0.1, psi2 = 1.0)

Monte = monte_carlo.Monte_Carlo(n_sample= n_sample, copula_sane= copula_sane, copula_contaminated = copula_contaminated, delta = 0.45)

sample, index = Monte.adversarial_contamination(inv_cdf_s= norm.ppf, inv_cdf_o=norm.ppf, show_index= True)

print(sample, index)

fig, ax = plt.subplots()
ax.plot(sample[index,0], sample[index,1], '.',markersize = 1, alpha = 0.75, color = 'lightcoral')
ax.plot(sample[~index,0], sample[~index,1], '.',markersize = 1, alpha = 0.5, color = 'lightblue')
ax.set_title(r'Adversarial contamination $\delta = 0.45$')
plt.savefig("/home/aboulin/Documents/stage/var_FMado/robust/output/test.pdf")

print(len(sample[index,:]))
