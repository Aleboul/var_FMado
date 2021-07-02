import extreme_value_copula
import monte_carlo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LMBD = np.linspace(0.01,0.99,100)
X = np.linspace(-0.3,0.3,100)

def gauss_function(x, sigma):
    return np.sqrt(1 / (2*np.pi * sigma**2)) * np.exp(-(x) ** 2 / (2 * sigma ** 2) )

theta = 1.0
psi1 = 1.0
psi2 = 1.0

copula = extreme_value_copula.Asy_log(copula_type = 27, theta = theta, psi1 = psi1, psi2 = psi2)
var_ = []

zs = []

for lmbd in LMBD:
    for x in X:
        sigma  = np.sqrt(copula.extreme_var_FMado(lmbd))
        value_ = gauss_function(x, sigma)
        zs.append(value_)
X, LMBD = np.meshgrid(X, LMBD)
zs = np.array(zs)
Z = zs.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, LMBD, Z)
ax.set_xlabel('X label')
ax.set_ylabel(r'$\lambda$')
ax.set_zlabel("Z value")
plt.savefig("/home/aboulin/Documents/stage/var_FMado/bivariate/output/image_3d.pdf")