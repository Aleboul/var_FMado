"""
    Produit l'expérience lorsque l'on prend le maximum de variables aléatoires suivant
    une gaussienne bivariée (liée à la Hüsler Reiss)
"""
import numpy as np
import pandas as pd
import monte_carlo
import extreme_value_copula
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
plt.style.use('seaborn-whitegrid')
from matplotlib import cm
n = 100
theta = 0.2


with open("output/max_student_M100_n128.txt", "rb") as data:
    b = pickle.load(data)

var_lmbd = pd.DataFrame(b)
x = np.linspace(0.01,0.99,n)
value = []
copula = extreme_value_copula.Student(theta = theta, psi1=3.0)
for lmbd in x:
    value_ = copula.var_FMado(lmbd)
    value.append(value_)

var_lmbd['var_theo'] = value

print(var_lmbd)

fig, ax = plt.subplots()
sns.scatterplot(data = var_lmbd, x = "lambda", y = "var_emp", hue = "nmax", palette= 'crest', s = 10, alpha = 0.5)
sns.lineplot(data = var_lmbd, x = "lambda", y = "var_theo", hue = "nmax", palette = 'crest',legend = False, lw = 1)
ax.legend(title = r'$\theta$')
ax.set_ylabel(r'$\sigma^2$')
ax.set_xlabel(r'$\lambda$')
plt.savefig("/home/aboulin/Documents/stage/var_FMado/bivariate/output/image_4.pdf")
