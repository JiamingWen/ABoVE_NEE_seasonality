'''plot evaluation on inversions' prior and posterior estimates'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

stat_var = 'cor'; xlim = [-0.2, 0.85]; xlabel = r'Correlation with CO$_{2}$ observations'
# stat_var = 'slope'; xlim = [-0.2, 2.2]; xlabel = r'Slope of regression with CO$_{2}$ observations'
# stat_var = 'intercept'; xlim = [-8, 2]; xlabel = r'Intercept of regression with CO$_{2}$ observations'
# stat_var = 'mean_bias'; xlim = [-8, 2]; xlabel = r'Mean bias compared to CO$_{2}$ observations'
# stat_var = 'rmse'; xlim = [0, 15]; xlabel = r'RMSE compared to CO$_{2}$ observations'


# all years
fitting_df_inversions_posterior = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}.csv')
fitting_df_inversions_posterior = fitting_df_inversions_posterior.loc[fitting_df_inversions_posterior['model_name'].isin(['CAMS', 'CMS-Flux', 'CTE', 'MIROC', 'NISMON-CO2']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_prior = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE-prior{lc_filestr}.csv')
fitting_df_inversions_prior = fitting_df_inversions_prior.loc[fitting_df_inversions_prior['model_name'].isin(['CAMS', 'CMS-Flux', 'CTE', 'MIROC', 'NISMON-CO2']), :] ## for models with no coverage of CARVE years
fig, ax = plt.subplots(figsize=(7,3))

# # only for 2017
# fitting_df_inversions_posterior = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_2017.csv') #_only_seasonal
# fitting_df_inversions_posterior = fitting_df_inversions_posterior.loc[fitting_df_inversions_posterior['model_name'].isin(['CAMS', 'CAMS-Satellite', 'CMS-Flux', 'CTE', 'GCASv2', 'GONGGA', 'MIROC', 'NISMON-CO2', 'THU']), :] ## for models with no coverage of CARVE years
# fitting_df_inversions_prior = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE-prior{lc_filestr}_2017.csv')
# fitting_df_inversions_prior = fitting_df_inversions_prior.loc[fitting_df_inversions_prior['model_name'].isin(['CAMS', 'CAMS-Satellite', 'CMS-Flux', 'CTE', 'GCASv2', 'GONGGA', 'MIROC', 'NISMON-CO2', 'THU']), :] ## for models with no coverage of CARVE years
# fig, ax = plt.subplots(figsize=(7,4))


fitting_df_inversions_merge = pd.merge(fitting_df_inversions_posterior, fitting_df_inversions_prior, on='model_name', how='outer', suffixes=('_posterior', '_prior'))
fitting_df_inversions_merge = fitting_df_inversions_merge.sort_values('model_name', ascending=False)

fitting_df_inversions_sorted = fitting_df_inversions_merge.sort_values('model_name', ascending=False)


p1 = plt.scatter(fitting_df_inversions_sorted[f'{stat_var}_posterior'], fitting_df_inversions_sorted['model_name'], marker='s', color='black', facecolor='none', s=60, label='posterior')
p2 = plt.scatter(fitting_df_inversions_sorted[f'{stat_var}_prior'], fitting_df_inversions_sorted['model_name'], marker='s', color='red', facecolor='none', s=60, label='prior')

plt.xlim(xlim)
plt.ylim(-1, fitting_df_inversions_sorted.shape[0]-0.5)
plt.xlabel(xlabel, fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=14)
plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_inversion_prior/Fig2_prior_{stat_var}.png', dpi=300, bbox_inches='tight')
plt.show()