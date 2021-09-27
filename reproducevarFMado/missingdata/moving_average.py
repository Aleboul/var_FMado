import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
from matplotlib import cm

df = pd.read_csv('/home/aboulin/Documents/stage/var_FMado/bivariate/output/zoom.csv')

ma = np.array(df.groupby(['theta']).rolling(window = 30, center = True).mean()['var_emp'])
df['moving_average'] = ma
print(df)

fig, ax = plt.subplots()
sns.lineplot(data = df, x = "lmbd", y = "moving_average", hue = "theta", palette= 'crest', lw = 1, linestyle = '--', alpha = 0.5)
sns.lineplot(data = df, x = "lmbd", y = "var_theo", hue = "theta", palette = 'crest',legend = False, lw = 1)
ax.legend(title = r'$\theta$')
ax.set_ylabel(r'$\sigma^2$')
ax.set_xlabel(r'$\lambda$')
plt.savefig("/home/aboulin/Documents/stage/var_FMado/bivariate/output/moving_average.pdf")