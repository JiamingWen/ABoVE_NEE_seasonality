'''plot an illustration of scatterplots from biased and reference NEE seasonality'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

option = 'cor' #none mean_bias range_ratio cor

np.random.seed(42)
x = np.linspace(-20, 20, 40)
noise = np.random.normal(loc=0.0, scale=2.0, size=x.shape)
y = x + noise

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(x, y, s=50, alpha=0.7, edgecolor='none')
if option == 'mean_bias':
    ax.scatter(x, y-15, s=50, alpha=0.7, edgecolor='none', facecolor='#d4631d')
elif option == 'range_ratio':
    ax.scatter(x, y*2.5, s=50, alpha=0.7, edgecolor='none', facecolor='#d4631d')
elif option == 'cor':
    ax.scatter(x, y/40*15 + 15 * np.sin(x / 7), s=50, alpha=0.7, edgecolor='none', facecolor='#d4631d')

ax.plot([-40, 40], [-40, 40], 'k--', linewidth=1)

plt.xlim(-40, 40)
plt.ylim(-40, 40)
plt.xticks([-40, -20, 0, 20, 40], fontsize=16)
plt.yticks([-40, -20, 0, 20, 40], fontsize=16)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

# Make math symbols large, add a smaller "(ppm)" as a separate annotation
ax.set_xlabel(r'$z_{\mathrm{obs}}$', fontsize=25)
ax.annotate('(ppm)', xy=(1.02, 0.4), xycoords=ax.xaxis.label, textcoords='offset points',
            xytext=(6, 0), ha='left', va='center', fontsize=16)

ax.set_ylabel(r'$z_{\mathrm{model}}$', fontsize=25)
ax.annotate('(ppm)', xy=(0.6, 1.02), xycoords=ax.yaxis.label, textcoords='offset points',
            xytext=(0, 6), ha='center', va='bottom', rotation=90, fontsize=16)

plt.tight_layout()
plt.show()