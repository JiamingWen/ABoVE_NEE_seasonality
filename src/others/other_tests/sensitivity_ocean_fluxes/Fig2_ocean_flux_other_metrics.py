'''
plot summary figure for model performance, e.g., mean bias, regression slopes/intercepts, RMSE
evaluation with removing the data filtering criterion of regional relavance
also accounting for CO2 enhancements due to ocean fluxes from data products' ensemble
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLSResults

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

# stat_var = 'cor'; xlim = [-0.1, 0.75]; xlabel = r'Correlation with CO$_{2}$ observations'
# stat_var = 'slope'; xlim = [-0.2, 2.2]; xlabel = r'Slope of regression with CO$_{2}$ observations'
# stat_var = 'intercept'; xlim = [-8, 2]; xlabel = r'Intercept of regression with CO$_{2}$ observations'
# stat_var = 'mean_bias'; xlim = [-8, 2]; xlabel = r'Mean bias compared to CO$_{2}$ observations'
stat_var = 'rmse'; xlim = [0, 15]; xlabel = r'RMSE compared to CO$_{2}$ observations'

# unscaled variables (without linear regression)
fitting_df_TRENDYv11_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}.csv')
# fitting_df_TRENDYv11_unscaled = fitting_df_TRENDYv11_unscaled.loc[~fitting_df_TRENDYv11_unscaled['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}.csv')
fitting_df_inversions_unscaled = fitting_df_inversions_unscaled.loc[~fitting_df_inversions_unscaled['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_unscaled.loc[fitting_df_inversions_unscaled['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_UpscaledEC_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}.csv')

# sort for each category
fitting_df_TRENDYv11_sorted = fitting_df_TRENDYv11_unscaled.sort_values('model_name', ascending=False)
fitting_df_inversions_sorted = fitting_df_inversions_unscaled.sort_values('model_name', ascending=False)
fitting_df_UpscaledEC_sorted = fitting_df_UpscaledEC_unscaled.sort_values('model_name', ascending=False)

# set colors
high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(high_skill_TRENDY),'color'] = '#396bb8'
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(low_skill_TRENDY),'color'] = '#d4631d'
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin (['IBIS']),'color'] = 'grey' ## models with negative Rh


fig, ax = plt.subplots(figsize=(7,10))

results = OLSResults.load(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/regression/TRENDYv11_CLM5.0{lc_filestr}.pickle")
n = results.summary2().tables[0].loc[3,1] # number of observations
labelname = f"Default\n(n = {n})"

plt.scatter(fitting_df_TRENDYv11_sorted[f'{stat_var}'], fitting_df_TRENDYv11_sorted['model_name'], marker='o', color='black', label=labelname, s=70, facecolors='none', linewidths=2) #, color=fitting_df_TRENDYv11_sorted['color']
plt.scatter(fitting_df_UpscaledEC_sorted[f'{stat_var}'], fitting_df_UpscaledEC_sorted['model_name'], marker='o', color='black', s=70, facecolors='none', linewidths=2) #, color='#6db753'
plt.scatter(fitting_df_inversions_sorted[f'{stat_var}'], fitting_df_inversions_sorted['model_name'], marker='o', color='black', s=70, facecolors='none', linewidths=2)  #, color='black'

plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')

plt.xlim(xlim)
plt.ylim(-1, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-0.5)
plt.xlabel(xlabel, fontsize=18)
plt.xticks(fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)


colors = fitting_df_TRENDYv11_sorted['color'].values.tolist() + ['#6db753']*fitting_df_UpscaledEC_sorted.shape[0] + ['black']*fitting_df_inversions_sorted.shape[0]
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)

ax.annotate("Atmospheric Inversions", (xlim[0]+(xlim[1]-xlim[0])/20, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-1.5), fontsize=15)
ax.annotate("Upscaled EC", (xlim[0]+(xlim[1]-xlim[0])/20, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]-1.3), fontsize=15)
ax.annotate("TRENDY TBMs", (xlim[0]+(xlim[1]-xlim[0])/20, fitting_df_TRENDYv11_sorted.shape[0]-1.5), fontsize=15)


##############################################################
# overlay with results for monthly outputs + monthly diurnal cycle from X-BASE
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

# unscaled variables (without linear regression)
fitting_df_TRENDYv11_unscaled_ocean_flux = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_ocean_flux.csv')
# fitting_df_TRENDYv11_unscaled_ocean_flux = fitting_df_TRENDYv11_unscaled_ocean_flux.loc[~fitting_df_TRENDYv11_unscaled_ocean_flux['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled_ocean_flux = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_ocean_flux.csv')
fitting_df_inversions_unscaled_ocean_flux = fitting_df_inversions_unscaled_ocean_flux.loc[~fitting_df_inversions_unscaled_ocean_flux['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_unscaled_ocean_flux.loc[fitting_df_inversions_unscaled_ocean_flux['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_UpscaledEC_unscaled_ocean_flux = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_ocean_flux.csv')

fitting_df_TRENDYv11_merge = pd.merge(fitting_df_TRENDYv11_sorted, fitting_df_TRENDYv11_unscaled_ocean_flux, on='model_name', how='outer', suffixes=('_default', '_ocean_flux'))
fitting_df_inversions_merge = pd.merge(fitting_df_inversions_sorted, fitting_df_inversions_unscaled_ocean_flux, on='model_name', how='outer', suffixes=('_default', '_ocean_flux'))
fitting_df_UpscaledEC_merge = pd.merge(fitting_df_UpscaledEC_sorted, fitting_df_UpscaledEC_unscaled_ocean_flux, on='model_name', how='outer', suffixes=('_default', '_ocean_flux'))


color = 'red'
labelname = f"Accounting for ocean fluxes\n(n = 85693)"

plt.scatter(fitting_df_TRENDYv11_merge[f'{stat_var}_ocean_flux'], fitting_df_TRENDYv11_merge['model_name'], marker='o', color=color, facecolors='none', label=labelname, s=70)
plt.scatter(fitting_df_inversions_merge[f'{stat_var}_ocean_flux'], fitting_df_inversions_merge['model_name'], marker='o', color=color, facecolors='none', s=70)
plt.scatter(fitting_df_UpscaledEC_merge[f'{stat_var}_ocean_flux'], fitting_df_UpscaledEC_merge['model_name'], marker='o', color=color, facecolors='none', s=70)

plt.legend(loc='best', fontsize=14)

fig.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_ocean_fluxes/Fig2_ocean_flux_other_metrics_{stat_var}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_ocean_fluxes/Fig2_ocean_flux_other_metrics_{stat_var}.pdf', dpi=300, bbox_inches='tight')
plt.show()