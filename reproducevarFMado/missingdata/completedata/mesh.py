"""
    Code permettant les représentations 3D de la variance asymptotique du processus.
    
    Inputs
    ------
    
    THETA : mesh grid for y-axis
    LMBD : mesh grid for x-axis
    psi1, psi2 : copula parameters
"""

import var_FMado.reproducevarFMado.missingdata.src.extreme_value_copula
import var_FMado.reproducevarFMado.missingdata.src.monte_carlo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LMBD = np.linspace(0.01,0.99,100)
THETA = np.linspace(-0.99,0.99,100)

def gauss_function(x, sigma):
    return np.sqrt(1 / (2*np.pi * sigma**2)) * np.exp(-(x) ** 2 / (2 * sigma ** 2) )
psi1 = 0.2
psi2 = 1.0
var_ = []

zs = []

for lmbd in LMBD:
    for theta in THETA: 
        copula = extreme_value_copula.Student(theta = theta, psi1 = psi1, psi2 = psi2)
        sigma  = copula.var_FMado(lmbd)
        zs.append(sigma)
    print(lmbd)
THETA, LMBD = np.meshgrid(THETA, LMBD)
zs = np.array(zs)
Z = zs.reshape(THETA.shape)
print(Z.shape)
print(THETA.shape)
print(LMBD.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(THETA, LMBD, Z, alpha = 0.5, color = 'darkblue')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\lambda$')
ax.set_zlabel(r"$\sigma^2$")
#ax.set_title('Student Model')
plt.savefig("/home/aboulin/Documents/stage/var_FMado/bivariate/output/Student_3d.pdf")
