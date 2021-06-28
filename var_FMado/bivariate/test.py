import frank
import monte_carlo
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib import cm

n = 50
n_iter = 150
n_sample = [64]
theta = -10
random_seed = 42

copula = frank.Frank(copula_type = 'FRANK', random_seed = 42, theta = theta, n_sample = np.max(n_sample))
Monte = monte_carlo.Monte_Carlo(n_iter= n_iter, n_sample= n_sample, random_seed= random_seed, copula= copula)
var_lmbd = Monte.exec_varlmbd(n)

x = np.linspace(0.01,0.99,n)
value = []

for lmbd in x:
    print(lmbd)
    value_ = copula.var_FMado(lmbd) 
    value.append(value_)

fig, ax = plt.subplots()
ax.plot(x, value, '--', color = 'darkblue')
ax.plot(x, var_lmbd, '.', markersize = 5, alpha = 0.5, color = 'salmon')
ax.set_ylabel(r'$\sigma^2$')
ax.set_xlabel(r'$\lambda$')
plt.savefig("/home/aboulin/Documents/stage/var_FMado/bivariate/output/image_3.png")