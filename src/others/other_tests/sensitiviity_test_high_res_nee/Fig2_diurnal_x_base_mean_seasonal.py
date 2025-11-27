'''
plot summary figure for model performance (correlation with observed CO2 enhancement)
evaluation for (1) monthly + monthly diurnal cycle from X-BASE and (2) monthly mean seasonal cycle + monthly diurnal cycle from X-BASE
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

stat_var = 'cor'; xlim = [-0.1, 0.75]

# unscaled variables (without linear regression)
fitting_df_TRENDYv11_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_diurnal_x_base.csv')
# fitting_df_TRENDYv11_unscaled = fitting_df_TRENDYv11_unscaled.loc[~fitting_df_TRENDYv11_unscaled['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_diurnal_x_base.csv')
fitting_df_inversions_unscaled = fitting_df_inversions_unscaled.loc[~fitting_df_inversions_unscaled['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_unscaled.loc[fitting_df_inversions_unscaled['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_UpscaledEC_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_diurnal_x_base.csv')


# in alphabetical order
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
p1a = plt.scatter(fitting_df_TRENDYv11_sorted[f'{stat_var}'], fitting_df_TRENDYv11_sorted['model_name'], marker='o', edgecolor=fitting_df_TRENDYv11_sorted['color'], facecolor='none', s=70)
p2a = plt.scatter(fitting_df_UpscaledEC_sorted[f'{stat_var}'], fitting_df_UpscaledEC_sorted['model_name'], marker='d', color='#56983f', facecolor='none', s=80)
p3a = plt.scatter(fitting_df_inversions_sorted[f'{stat_var}'], fitting_df_inversions_sorted['model_name'], marker='s', color='black', facecolor='none', s=60)

# section lines
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')

plt.xlim(xlim)
plt.ylim(-1, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]-0.5) 
plt.xlabel(r'Correlation with CO$_{2}$ observations', fontsize=18)
plt.xticks(ticks=np.arange(xlim[0], xlim[1], 0.1), fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)


colors = fitting_df_TRENDYv11_sorted['color'].values.tolist() + ['#56983f']*fitting_df_UpscaledEC_sorted.shape[0] + ['black']*fitting_df_inversions_sorted.shape[0]
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)

ax.annotate("Atmospheric Inversions", (-0.08, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-1.5), fontsize=15)
ax.annotate("Upscaled EC", (-0.08, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]-1.3), fontsize=15)
ax.annotate("TRENDY TBMs", (-0.08, fitting_df_TRENDYv11_sorted.shape[0]-2.5), fontsize=15)


##########################################################################################
# add additional evaluation with mean seasonal cycle
fitting_df_TRENDYv11_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_only_seasonal_diurnal_x_base.csv')
# fitting_df_TRENDYv11_unscaled_only_seasonal = fitting_df_TRENDYv11_unscaled_only_seasonal.loc[~fitting_df_TRENDYv11_unscaled_only_seasonal['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_only_seasonal_diurnal_x_base.csv')
fitting_df_inversions_unscaled_only_seasonal = fitting_df_inversions_unscaled_only_seasonal.loc[~fitting_df_inversions_unscaled_only_seasonal['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_unscaled_only_seasonal.loc[fitting_df_inversions_unscaled_only_seasonal['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_UpscaledEC_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_only_seasonal_diurnal_x_base.csv')


fitting_df_TRENDYv11_merge = pd.merge(fitting_df_TRENDYv11_sorted, fitting_df_TRENDYv11_unscaled_only_seasonal, on='model_name', how='outer', suffixes=('_original', '_only_seasonal'))
fitting_df_inversions_merge = pd.merge(fitting_df_inversions_sorted, fitting_df_inversions_unscaled_only_seasonal, on='model_name', how='outer', suffixes=('_original', '_only_seasonal'))
fitting_df_UpscaledEC_merge = pd.merge(fitting_df_UpscaledEC_sorted, fitting_df_UpscaledEC_unscaled_only_seasonal, on='model_name', how='outer', suffixes=('_original', '_only_seasonal'))

p1b = plt.scatter(fitting_df_TRENDYv11_merge[f'{stat_var}_only_seasonal'], fitting_df_TRENDYv11_merge['model_name'], marker='o', color=fitting_df_TRENDYv11_sorted['color'], s=70) #, alpha=0.8
p2b = plt.scatter(fitting_df_UpscaledEC_merge[f'{stat_var}_only_seasonal'], fitting_df_UpscaledEC_merge['model_name'], marker='d', color='#56983f', s=80)
p3b = plt.scatter(fitting_df_inversions_merge[f'{stat_var}_only_seasonal'], fitting_df_inversions_merge['model_name'], marker='s', color='black', s=60) #, label='Mean seasonal cycle'

plt.legend([(p1a, p2a, p3a), (p1b, p2b, p3b)], ['Original data', 'Mean seasonal cycle'],
           bbox_to_anchor=(0.65, 0.7), fontsize=14, handletextpad=1.5, scatterpoints=1, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None, pad=2)}, frameon=True, borderpad=1, labelspacing=0.5)

fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/Fig2_diurnal_x_base_mean_seasonal.png', dpi=300, bbox_inches='tight')
fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/Fig2_diurnal_x_base_mean_seasonal.pdf', dpi=300, bbox_inches='tight')
plt.show()