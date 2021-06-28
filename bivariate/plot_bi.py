import nelsen
import monte_carlo
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib import cm

n_sample = [1000]
theta = 1
random_seed = 42
copula = nelsen.Nelsen_9(copula_type = 'NELSEN_9', random_seed = 42, theta = theta, n_sample = np.max(n_sample))
sampeul = copula.sample_unimargin()
print(sampeul)

fig, ax = plt.subplots()
ax.plot(sampeul[:,0], sampeul[:,1], '.',markersize = 5, alpha = 0.5, color = 'salmon')
plt.savefig("/home/aboulin/Documents/stage/var_FMado/bivariate/output/image_1.pdf")