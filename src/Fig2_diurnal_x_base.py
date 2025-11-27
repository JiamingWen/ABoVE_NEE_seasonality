'''
plot summary figure for model performance (correlation with observed CO2 enhancement)
evaluation for monthly vs monthly + monthly diurnal cycle from X-BASE
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

stat_var = 'cor'; xlim = [-0.2, 0.85]

# unscaled variables (without linear regression)
fitting_df_TRENDYv11_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}.csv')
# fitting_df_TRENDYv11_unscaled = fitting_df_TRENDYv11_unscaled.loc[~fitting_df_TRENDYv11_unscaled['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}.csv')
fitting_df_inversions_unscaled = fitting_df_inversions_unscaled.loc[~fitting_df_inversions_unscaled['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_unscaled.loc[fitting_df_inversions_unscaled['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_UpscaledEC_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}.csv')

# sort for each category
fitting_df_TRENDYv11_sorted = fitting_df_TRENDYv11_unscaled.sort_values(f'{stat_var}')
fitting_df_inversions_sorted = fitting_df_inversions_unscaled.sort_values(f'{stat_var}')
fitting_df_UpscaledEC_sorted = fitting_df_UpscaledEC_unscaled.sort_values(f'{stat_var}')

# set colors
high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(high_skill_TRENDY),'color'] = '#396bb8'
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(low_skill_TRENDY),'color'] = '#d4631d'
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin (['IBIS']),'color'] = 'grey' ## models with negative Rh


fig, ax = plt.subplots(figsize=(7,10))

labelname = f"Monthly"

plt.scatter(fitting_df_TRENDYv11_sorted[f'{stat_var}'], fitting_df_TRENDYv11_sorted['model_name'], marker='o', color='black', label=labelname, s=70, facecolors='none', linewidths=2) #, color=fitting_df_TRENDYv11_sorted['color']
plt.scatter(fitting_df_UpscaledEC_sorted[f'{stat_var}'], fitting_df_UpscaledEC_sorted['model_name'], marker='o', color='black', s=70, facecolors='none', linewidths=2) #, color='#6db753'
plt.scatter(fitting_df_inversions_sorted[f'{stat_var}'], fitting_df_inversions_sorted['model_name'], marker='o', color='black', s=70, facecolors='none', linewidths=2)  #, color='black'

plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')

plt.xlim(xlim)
plt.ylim(-1, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-0.5)
plt.xlabel(r'Correlation with CO$_{2}$ observations', fontsize=18)
plt.xticks(ticks=np.arange(-0.2, 0.9, 0.1), fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)


colors = fitting_df_TRENDYv11_sorted['color'].values.tolist() + ['#6db753']*fitting_df_UpscaledEC_sorted.shape[0] + ['black']*fitting_df_inversions_sorted.shape[0]
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)

plt.annotate("Atmospheric Inversions", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-1.5), fontsize=15)
plt.annotate("Upscaled EC", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]-1.3), fontsize=15)
plt.annotate("TRENDY", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]-1.5), fontsize=15)


##############################################################
# overlay with results for monthly outputs + monthly diurnal cycle from X-BASE
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

# unscaled variables (without linear regression)
fitting_df_TRENDYv11_unscaled_diurnal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_diurnal_x_base.csv')
# fitting_df_TRENDYv11_unscaled_diurnal = fitting_df_TRENDYv11_unscaled_diurnal.loc[~fitting_df_TRENDYv11_unscaled_diurnal['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled_diurnal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_diurnal_x_base.csv')
fitting_df_inversions_unscaled_diurnal = fitting_df_inversions_unscaled_diurnal.loc[~fitting_df_inversions_unscaled_diurnal['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_unscaled_diurnal.loc[fitting_df_inversions_unscaled_diurnal['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_UpscaledEC_unscaled_diurnal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_diurnal_x_base.csv')


fitting_df_TRENDYv11_merge = pd.merge(fitting_df_TRENDYv11_sorted, fitting_df_TRENDYv11_unscaled_diurnal, on='model_name', how='outer', suffixes=('_monthly', '_diurnal_x_base'))
fitting_df_inversions_merge = pd.merge(fitting_df_inversions_sorted, fitting_df_inversions_unscaled_diurnal, on='model_name', how='outer', suffixes=('_monthly', '_diurnal_x_base'))
fitting_df_UpscaledEC_merge = pd.merge(fitting_df_UpscaledEC_sorted, fitting_df_UpscaledEC_unscaled_diurnal, on='model_name', how='outer', suffixes=('_monthly', '_diurnal_x_base'))


color = 'red' # color for the case with diurnal cycle
labelname = f"Monthly + X-BASE diurnal cycle"

plt.scatter(fitting_df_TRENDYv11_merge[f'{stat_var}_diurnal_x_base'], fitting_df_TRENDYv11_merge['model_name'], marker='o', color=color, facecolors='none', label=labelname, s=70)
plt.scatter(fitting_df_inversions_merge[f'{stat_var}_diurnal_x_base'], fitting_df_inversions_merge['model_name'], marker='o', color=color, facecolors='none', s=70)
plt.scatter(fitting_df_UpscaledEC_merge[f'{stat_var}_diurnal_x_base'], fitting_df_UpscaledEC_merge['model_name'], marker='o', color=color, facecolors='none', s=70)

plt.legend(bbox_to_anchor=(0.75, 0.8), fontsize=14)

fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig2_diurnal_x_base.png', dpi=300, bbox_inches='tight')
fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig2_diurnal_x_base.pdf', dpi=300, bbox_inches='tight')
plt.show()