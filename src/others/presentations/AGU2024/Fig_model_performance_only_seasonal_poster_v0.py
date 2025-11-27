# plot summary figure for model performance (correlation with observed CO2 enhancement)
# removed regression models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/src')
from statsmodels.regression.linear_model import OLSResults
from scipy import stats

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

stat_var = 'cor'; xlim = [-0.1, 0.75]

# unscaled variables (without linear regression)
fitting_df_TRENDYv11_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_TRENDYv11{lc_filestr}.csv')
# fitting_df_TRENDYv11_unscaled = fitting_df_TRENDYv11_unscaled.loc[~fitting_df_TRENDYv11_unscaled['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_inversions{lc_filestr}.csv')
fitting_df_inversions_unscaled = fitting_df_inversions_unscaled.loc[~fitting_df_inversions_unscaled['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_NEEobservations_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_NEEobservations{lc_filestr}.csv')
fitting_df_NEEobservations_unscaled.loc[fitting_df_NEEobservations_unscaled['model_name'] == 'FluxCOM-X-NEE','model_name'] = 'X-BASE NEE'

# scaled variables (with linear regression)
fitting_df_reference_scaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_reference{lc_filestr}.csv')
fitting_df_reference_scaled = fitting_df_reference_scaled.loc[fitting_df_reference_scaled['model_name'].isin(['APAR', 'PAR', 'FPAR', 'GOME2_SIF']), :]  #'APAR', 'FPAR', 'LAI', 'PAR'
fitting_df_reference_scaled.loc[fitting_df_reference_scaled['model_name'] == 'GOME2_SIF','model_name'] = 'SIF'

# sort for each category
fitting_df_TRENDYv11_sorted = fitting_df_TRENDYv11_unscaled.sort_values(f'{stat_var}')
fitting_df_inversions_sorted = fitting_df_inversions_unscaled.sort_values(f'{stat_var}')
fitting_df_reference_sorted = fitting_df_reference_scaled.sort_values(f'{stat_var}')

# set colors
high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(high_skill_TRENDY),'color'] = '#5986cb'
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(low_skill_TRENDY),'color'] = '#e57f3f'
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin (['IBIS']),'color'] = '#5986cb' #'grey' ## models with negative Rh

fig, ax = plt.subplots(figsize=(7,10))
plt.scatter(fitting_df_TRENDYv11_sorted[f'{stat_var}'], fitting_df_TRENDYv11_sorted['model_name'], marker='o', color=fitting_df_TRENDYv11_sorted['color'], s=50)
plt.scatter(fitting_df_inversions_sorted[f'{stat_var}'], fitting_df_inversions_sorted['model_name'], marker='s', color='black', s=50)
plt.scatter(fitting_df_NEEobservations_unscaled[f'{stat_var}'], fitting_df_NEEobservations_unscaled['model_name'], marker='d', color='#6db753', s=80)
plt.scatter(fitting_df_reference_sorted[f'{stat_var}'], fitting_df_reference_sorted['model_name'], marker='v', color='purple', s=80)

plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_NEEobservations_unscaled.shape[0]+fitting_df_reference_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.xlim(xlim)
plt.ylim(-1, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_NEEobservations_unscaled.shape[0]+fitting_df_reference_sorted.shape[0]-0.5)
plt.xlabel(r'Correlation with CO$_{2}$ observations', fontsize=18)
plt.xticks(ticks=np.arange(-0.1, 0.8, 0.1), fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)

# results = OLSResults.load(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/TRENDYv11_CLM5.0{lc_filestr}.pickle")
# n = results.summary2().tables[0].loc[3,1] # number of observations
# # plt.title(f'{lcname}(n={n})', fontsize=20)

colors = fitting_df_TRENDYv11_sorted['color'].values.tolist() + ['black']*fitting_df_inversions_sorted.shape[0] + ['#6db753']*fitting_df_NEEobservations_unscaled.shape[0] + ['purple']*fitting_df_reference_sorted.shape[0]
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)

plt.annotate("Remote Sensing", (-0.05, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_NEEobservations_unscaled.shape[0]+fitting_df_reference_sorted.shape[0]-1.5), fontsize=15)
plt.annotate("Upscaled EC flux", (-0.05, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_NEEobservations_unscaled.shape[0]-1.3), fontsize=15)
plt.annotate("Inversions", (-0.05, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-1.5), fontsize=15)
plt.annotate("TRENDY", (-0.05, fitting_df_TRENDYv11_sorted.shape[0]-1.5), fontsize=15)



# add additional evaluation with mean seasonal cycle
    
fitting_df_TRENDYv11_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_only_seasonal.csv')
# fitting_df_TRENDYv11_unscaled_only_seasonal = fitting_df_TRENDYv11_unscaled_only_seasonal.loc[~fitting_df_TRENDYv11_unscaled_only_seasonal['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_only_seasonal.csv')
fitting_df_inversions_unscaled_only_seasonal = fitting_df_inversions_unscaled_only_seasonal.loc[~fitting_df_inversions_unscaled_only_seasonal['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_NEEobservations_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_NEEobservations{lc_filestr}_only_seasonal.csv')
fitting_df_NEEobservations_unscaled_only_seasonal.loc[fitting_df_NEEobservations_unscaled_only_seasonal['model_name'] == 'FluxCOM-X-NEE','model_name'] = 'X-BASE NEE'
fitting_df_reference_scaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_reference_only_seasonal{lc_filestr}.csv')
fitting_df_reference_scaled_only_seasonal = fitting_df_reference_scaled_only_seasonal.loc[fitting_df_reference_scaled_only_seasonal['model_name'].isin(['APAR', 'PAR', 'FPAR', 'GOME2_SIF']), :]
fitting_df_reference_scaled_only_seasonal.loc[fitting_df_reference_scaled_only_seasonal['model_name'] == 'GOME2_SIF','model_name'] = 'SIF'

fitting_df_TRENDYv11_merge = pd.merge(fitting_df_TRENDYv11_sorted, fitting_df_TRENDYv11_unscaled_only_seasonal, on='model_name', how='outer', suffixes=('_original', '_only_seasonal'))
fitting_df_inversions_merge = pd.merge(fitting_df_inversions_sorted, fitting_df_inversions_unscaled_only_seasonal, on='model_name', how='outer', suffixes=('_original', '_only_seasonal'))
fitting_df_reference_merge = pd.merge(fitting_df_reference_sorted, fitting_df_reference_scaled_only_seasonal, on='model_name', how='outer', suffixes=('_original', '_only_seasonal'))


plt.scatter(fitting_df_TRENDYv11_merge[f'{stat_var}_only_seasonal'], fitting_df_TRENDYv11_merge['model_name'], marker='x', color=fitting_df_TRENDYv11_sorted['color'], s=80)
plt.scatter(fitting_df_inversions_merge[f'{stat_var}_only_seasonal'], fitting_df_inversions_merge['model_name'], marker='x', color='black', s=80, label='Mean seasonal cycle')
plt.scatter(fitting_df_NEEobservations_unscaled_only_seasonal[f'{stat_var}'], fitting_df_NEEobservations_unscaled_only_seasonal['model_name'], marker='x', color='#6db753', s=80)
plt.scatter(fitting_df_reference_merge[f'{stat_var}_only_seasonal'], fitting_df_reference_merge['model_name'], marker='x', color='purple', s=80)

plt.axvspan(fitting_df_reference_scaled_only_seasonal.loc[fitting_df_reference_scaled_only_seasonal['model_name']=='APAR','cor_CI_low'].values[0], fitting_df_reference_scaled_only_seasonal.loc[fitting_df_reference_scaled_only_seasonal['model_name']=='APAR','cor_CI_high'].values[0], alpha=0.2, color='purple')

plt.legend(bbox_to_anchor=(0.45, 0.25), fontsize=12, handletextpad=0.2)

plt.show()
fig.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/figures/Fig_model_performance_only_seasonal{lc_filestr}_poster.png', bbox_inches='tight', dpi=300)
