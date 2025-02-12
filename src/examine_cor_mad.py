'''
Analyze the relationship between NEE seasonlity bias and consistency with atmospheric observations
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

reference = 'obs' # GroupH obs

fitting_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonality_diff/seasonal_mad_{reference}_NEE_ABoVEcore_unweighted.csv')

high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #
upscaled_EC = ['X-BASE', 'ABCflux']
# set colors
fitting_df.loc[fitting_df['model_name'].isin(high_skill_TRENDY),'color'] = '#396bb8'
fitting_df.loc[fitting_df['model_name'].isin(low_skill_TRENDY),'color'] = '#d4631d'
fitting_df.loc[fitting_df['model_name'].isin (['IBIS']),'color'] = 'grey' ## models with negative Rh
fitting_df.loc[fitting_df['model_name'].isin(upscaled_EC),'color'] = '#56983f'
# set shapes
fitting_df.loc[fitting_df['model_name'].isin(high_skill_TRENDY + low_skill_TRENDY + ['IBIS']),'shape'] = 'o'
fitting_df.loc[fitting_df['model_name'].isin(upscaled_EC),'shape'] = 'd'

fig, ax = plt.subplots(figsize=(5,5))
for i in np.arange(fitting_df.shape[0]):
    plt.scatter(fitting_df.loc[i, 'cor'], fitting_df.loc[i, 'mean_seasonal_diff'],color=fitting_df.loc[i, 'color'], marker=fitting_df.loc[i, 'shape'], s=50)

# add a regression line
x = fitting_df['cor']
y = fitting_df['mean_seasonal_diff']
m, b = np.polyfit(x, y, 1)
x0 = np.arange(-0.1,0.9,0.1)
plt.plot(x0, m * x0 + b, color='k', linestyle='--', alpha=0.8)
corr, _ = pearsonr(x, y)
plt.text(0.5, 0.55, f'Cor: {"{:.2f}".format(corr)}', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'Correlation with observed CO$_{2}$ enhancement', fontsize=13)
plt.ylabel('MAD of rescaled seasonal cycle', fontsize=13)

fig.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/cor_mad_{reference}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/cor_mad_{reference}.pdf', dpi=300, bbox_inches='tight')

plt.show()
