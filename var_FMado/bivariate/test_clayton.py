import nelsen
import monte_carlo
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib import cm
from scipy.integrate import quad, dblquad

n = 10
n_iter = 150
n_sample = [64]
theta = 1
random_seed = 42

copula = nelsen.Nelsen_9(copula_type = 'NELSEN_9', random_seed = 42, theta = theta, n_sample = np.max(n_sample))
print(copula.var_FMado(0.1))
