'''
Analyze the relationship between mean NEE and mean bias with atmospheric observations
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

background = '_background-ebg' # '' '_background-ct' '_background-ebg'
seasonal = '' # '' _only_seasonal

# statistical metrics
fitting_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{seasonal}{background}.csv')
fitting_df_upscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{seasonal}{background}.csv')
fitting_df_inversionsNEE = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{seasonal}{background}.csv')
fitting_df = pd.concat([fitting_df_TRENDYv11, fitting_df_upscaledEC, fitting_df_inversionsNEE], ignore_index=True)

# mean NEE
seasonal_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11_ABoVEcore_alllc_unweighted.csv')
seasonal_df_upscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_UpscaledEC_ABoVEcore_alllc_unweighted.csv')
seasonal_df_inversionsNEE = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_inversionsNEE_ABoVEcore_alllc_unweighted.csv')
seasonal_df = pd.concat([seasonal_df_TRENDYv11, seasonal_df_upscaledEC, seasonal_df_inversionsNEE], axis=1)
seasonal_mean = seasonal_df.iloc[3:11, :].mean() # seasonal mean for Apr - Nov

# merge the two dataframes
seasonal_mean = seasonal_mean.rename_axis('model_name').reset_index()
fitting_df = pd.merge(fitting_df, seasonal_mean[['model_name', 0]], on='model_name', how='left')
fitting_df = fitting_df.rename(columns={0: 'seasonal_mean'})

high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #
upscaled_EC = ['X-BASE', 'ABCflux']
inversions = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE']

# set colors
fitting_df.loc[fitting_df['model_name'].isin(high_skill_TRENDY),'color'] = '#396bb8'
fitting_df.loc[fitting_df['model_name'].isin(low_skill_TRENDY),'color'] = '#d4631d'
fitting_df.loc[fitting_df['model_name'].isin (['IBIS']),'color'] = 'grey' ## models with negative Rh
fitting_df.loc[fitting_df['model_name'].isin(upscaled_EC),'color'] = '#56983f'
fitting_df.loc[fitting_df['model_name'].isin(inversions),'color'] = 'black'
# set shapes
fitting_df.loc[fitting_df['model_name'].isin(high_skill_TRENDY + low_skill_TRENDY + ['IBIS']),'shape'] = 'o'
fitting_df.loc[fitting_df['model_name'].isin(upscaled_EC),'shape'] = 'd'
fitting_df.loc[fitting_df['model_name'].isin(inversions),'shape'] = 's'

# # exclude LPX-Bern
# fitting_df = fitting_df[~fitting_df['model_name'].isin(['LPX-Bern'])].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5,5))
for i in np.arange(fitting_df.shape[0]):
    plt.scatter(fitting_df.loc[i, 'mean_bias'], fitting_df.loc[i, 'seasonal_mean'], marker=fitting_df.loc[i, 'shape'], s=50, facecolor='none', edgecolor=fitting_df.loc[i, 'color']) #,color=fitting_df.loc[i, 'color']

# add a regression line
x = fitting_df['mean_bias']
y = fitting_df['seasonal_mean']
m, b = np.polyfit(x, y, 1)
x0 = np.arange(-9,2,0.1)
plt.plot(x0, m * x0 + b, color='k', linestyle='--', alpha=0.8)
corr, _ = pearsonr(x, y)
plt.text(0.95, 0.05, f'Cor: {"{:.2f}".format(corr)}', fontsize=15, ha='right', va='bottom', transform=ax.transAxes)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'Mean bias compared to CO$_{2}$ observations', fontsize=13)
plt.ylabel('Growing season mean NEE '+ '($\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=13)
plt.axvline(x=0, color='gray', linestyle=':', linewidth=1.5)

fig.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/mean_bias_mean_NEE{background}{seasonal}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/mean_bias_mean_NEE{background}{seasonal}.pdf', dpi=300, bbox_inches='tight')

plt.show()
