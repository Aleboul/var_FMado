"""
    Mesure la performance entre le FMadogramme et le MoN-FMadogramme
    
    Inputs
    ------
    n : grid's length of delta (proportion of outliers)
    n_sample : sample's length
    n_iter : number of estimator used for computing the biais
    theta, psi1, psi2 : copula's parameters
    copula_sane : copule of sane observations
    sample_outliers : design of the outliers
"""

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
import math

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

n = 100
n_sample = [1000]
n_iter = 100
theta = 1.5
psi1 = 1.0
psi2 = 1.0
copula_sane = extreme_value_copula.Asy_log(theta = theta, psi1= psi1, psi2= psi2)

delta_ = np.linspace(0.45,0.5, num = n)
q_05, q_95, q_50 = np.zeros(n), np.zeros(n), np.zeros(n)

q_05_MoN, q_95_MoN, q_50_MoN = np.zeros(n), np.zeros(n), np.zeros(n)
for i, delta in enumerate(delta_):
    
    K = math.ceil((0.5 - delta) * np.max(n_sample) + 1)
    Monte = monte_carlo.Monte_Carlo(n_iter = n_iter, n_sample= n_sample, copula_sane= copula_sane, sample_outliers= sample_outliers,delta = delta, K = K)
    df_FMado = Monte.simu(inv_cdf_s= norm.ppf, contamination= "Huber")

    q_05[i], q_95[i], q_50[i] = np.power(df_FMado['scaled'], 2).quantile(0.05), np.power(df_FMado['scaled'], 2).quantile(0.95), np.power(df_FMado['scaled'], 2).quantile(0.5)
    q_05_MoN[i], q_95_MoN[i], q_50_MoN[i] = np.power(df_FMado['scaled_MoN'], 2).quantile(0.05), np.power(df_FMado['scaled_MoN'], 2).quantile(0.95), np.power(df_FMado['scaled_MoN'], 2).quantile(0.5)

output = np.c_[q_05,q_50,q_95,q_05_MoN,q_50_MoN,q_95_MoN, delta_]
quantiles = pd.DataFrame(output)
quantiles.columns = ['q_05', 'q_50', "q_95", "q_05_MoN", "q_50_MoN", "q_95_MoN", "delta"]
print(quantiles)

fig, ax = plt.subplots()
ax.plot(quantiles.delta,quantiles.q_50, label = r"$\hat{\nu}$")
ax.fill_between(quantiles.delta, quantiles.q_05, quantiles.q_95, alpha =0.3)

ax.plot(quantiles.delta,quantiles.q_50_MoN, label = r'$\hat{\nu}_{MoN}$')
ax.fill_between(quantiles.delta, quantiles.q_05_MoN, quantiles.q_95_MoN, alpha =0.3)

ax.set_ylabel(r'$(\hat{\nu} - \nu)^2$')
ax.set_xlabel(r'$\delta$')
ax.legend(loc = 'best')
plt.savefig("/home/aboulin/Documents/stage/var_FMado/robust/output/gumbel_huber_top_left.pdf")
