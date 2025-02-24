'''
plot summary figure for model performance (correlation with observed CO2 enhancement)
evaluation separately for each land cover
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
fitting_df_NEEobservations_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_NEEobservations{lc_filestr}.csv')
fitting_df_NEEobservations_unscaled.loc[fitting_df_NEEobservations_unscaled['model_name'] == 'FluxCOM-X-NEE','model_name'] = 'X-BASE NEE'
fitting_df_NEEobservations_unscaled.loc[fitting_df_NEEobservations_unscaled['model_name'] == 'ABCflux-NEE','model_name'] = 'ABCflux'

# scaled variables (with linear regression)
fitting_df_reference_scaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_scaled_reference{lc_filestr}.csv')
fitting_df_reference_scaled = fitting_df_reference_scaled.loc[fitting_df_reference_scaled['model_name'].isin(['APAR', 'GOME2_SIF']), :]  #'APAR', 'FPAR', 'LAI', 'PAR'
fitting_df_reference_scaled.loc[fitting_df_reference_scaled['model_name'] == 'GOME2_SIF','model_name'] = 'SIF'

# sort for each category
fitting_df_TRENDYv11_sorted = fitting_df_TRENDYv11_unscaled.sort_values(f'{stat_var}')
fitting_df_inversions_sorted = fitting_df_inversions_unscaled.sort_values(f'{stat_var}')
fitting_df_NEEobservations_sorted = fitting_df_NEEobservations_unscaled.sort_values(f'{stat_var}')
fitting_df_reference_sorted = fitting_df_reference_scaled.sort_values(f'{stat_var}')

# set colors
high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(high_skill_TRENDY),'color'] = '#396bb8'
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(low_skill_TRENDY),'color'] = '#d4631d'
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin (['IBIS']),'color'] = 'grey' ## models with negative Rh


fig, ax = plt.subplots(figsize=(7,10))

results = OLSResults.load(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/regression/TRENDYv11_CLM5.0{lc_filestr}.pickle")
n = results.summary2().tables[0].loc[3,1] # number of observations
# plt.title(f'{lcname}(n={n})', fontsize=20)
lcname_new_dict = {"alllc":"All", 'forest':'Forests', 'shrub':'Shrubs', 'tundra':'Tundra'}
labelname = f"{lcname_new_dict[lcname]} (n={n})"

plt.scatter(fitting_df_TRENDYv11_sorted[f'{stat_var}'], fitting_df_TRENDYv11_sorted['model_name'], marker='o', color='black', label=labelname, s=70, facecolors='none', linewidths=2) #, color=fitting_df_TRENDYv11_sorted['color']
plt.scatter(fitting_df_NEEobservations_sorted[f'{stat_var}'], fitting_df_NEEobservations_sorted['model_name'], marker='o', color='black', s=70, facecolors='none', linewidths=2) #, color='#6db753'
plt.scatter(fitting_df_inversions_sorted[f'{stat_var}'], fitting_df_inversions_sorted['model_name'], marker='o', color='black', s=70, facecolors='none', linewidths=2)  #, color='black'
plt.scatter(fitting_df_reference_sorted[f'{stat_var}'], fitting_df_reference_sorted['model_name'], marker='o', color='black', s=70, facecolors='none', linewidths=2)  #, color='purple'

plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')

plt.xlim(xlim)
plt.ylim(-1, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_reference_sorted.shape[0]-0.5)
plt.xlabel(r'Correlation with CO$_{2}$ observations', fontsize=18)
plt.xticks(ticks=np.arange(-0.2, 0.9, 0.1), fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)


colors = fitting_df_TRENDYv11_sorted['color'].values.tolist() + ['#6db753']*fitting_df_NEEobservations_sorted.shape[0] + ['black']*fitting_df_inversions_sorted.shape[0] + ['purple']*fitting_df_reference_sorted.shape[0]
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)

plt.annotate("Remote Sensing", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_reference_sorted.shape[0]-1.5), fontsize=15)
plt.annotate("Atmospheric Inversions", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-1.5), fontsize=15)
plt.annotate("Upscaled EC", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_sorted.shape[0]-1.3), fontsize=15)
plt.annotate("TRENDY", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]-1.5), fontsize=15)


##############################################################
# overlay with results for different land covers
for lcname, color in zip(['forest', 'shrub', 'tundra'], ["#05450a", "#c6b044", "#ffafdc"]):

    if lcname == 'alllc':
        lc_filestr = ''
    elif lcname in ['forest', 'shrub', 'tundra']:
        lc_filestr = '_' + lcname

    
    # unscaled variables (without linear regression)
    fitting_df_TRENDYv11_unscaled_lc = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}.csv')
    # fitting_df_TRENDYv11_unscaled_lc = fitting_df_TRENDYv11_unscaled_lc.loc[~fitting_df_TRENDYv11_unscaled_lc['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
    fitting_df_inversions_unscaled_lc = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}.csv')
    fitting_df_inversions_unscaled_lc = fitting_df_inversions_unscaled_lc.loc[~fitting_df_inversions_unscaled_lc['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
    fitting_df_inversions_unscaled_lc.loc[fitting_df_inversions_unscaled_lc['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
    fitting_df_NEEobservations_unscaled_lc = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_NEEobservations{lc_filestr}.csv')
    fitting_df_NEEobservations_unscaled_lc.loc[fitting_df_NEEobservations_unscaled_lc['model_name'] == 'FluxCOM-X-NEE','model_name'] = 'X-BASE NEE'
    fitting_df_NEEobservations_unscaled.loc[fitting_df_NEEobservations_unscaled['model_name'] == 'ABCflux-NEE','model_name'] = 'ABCflux'

    # scaled variables (with linear regression)
    fitting_df_reference_scaled_lc = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_scaled_reference{lc_filestr}.csv')
    fitting_df_reference_scaled_lc = fitting_df_reference_scaled_lc.loc[fitting_df_reference_scaled_lc['model_name'].isin(['APAR', 'GOME2_SIF']), :]  #'APAR', 'FPAR', 'LAI', 'PAR'
    fitting_df_reference_scaled_lc.loc[fitting_df_reference_scaled_lc['model_name'] == 'GOME2_SIF','model_name'] = 'SIF'


    fitting_df_TRENDYv11_merge = pd.merge(fitting_df_TRENDYv11_sorted, fitting_df_TRENDYv11_unscaled_lc, on='model_name', how='outer', suffixes=('_all', '_lc'))
    fitting_df_inversions_merge = pd.merge(fitting_df_inversions_sorted, fitting_df_inversions_unscaled_lc, on='model_name', how='outer', suffixes=('_all', '_lc'))
    fitting_df_NEEobservations_merge = pd.merge(fitting_df_NEEobservations_sorted, fitting_df_NEEobservations_unscaled_lc, on='model_name', how='outer', suffixes=('_all', '_lc'))
    fitting_df_reference_merge = pd.merge(fitting_df_reference_scaled, fitting_df_reference_scaled_lc, on='model_name', how='outer', suffixes=('_all', '_lc'))


    results = OLSResults.load(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/regression/TRENDYv11_CLM5.0{lc_filestr}.pickle")
    n = results.summary2().tables[0].loc[3,1] # number of observations
    lcname_new_dict = {"alllc":"All", 'forest':'Forests', 'shrub':'Shrubs', 'tundra':'Tundra'}
    labelname = f"{lcname_new_dict[lcname]} (n={n})"
    
    plt.scatter(fitting_df_TRENDYv11_merge[f'{stat_var}_lc'], fitting_df_TRENDYv11_merge['model_name'], marker='o', color=color, facecolors='none', label=labelname, s=70)
    plt.scatter(fitting_df_inversions_merge[f'{stat_var}_lc'], fitting_df_inversions_merge['model_name'], marker='o', color=color, facecolors='none', s=70)
    plt.scatter(fitting_df_NEEobservations_merge[f'{stat_var}_lc'], fitting_df_NEEobservations_merge['model_name'], marker='o', color=color, facecolors='none', s=70)
    plt.scatter(fitting_df_reference_merge[f'{stat_var}_lc'], fitting_df_reference_merge['model_name'], marker='o', color=color, facecolors='none', s=70)

plt.legend(bbox_to_anchor=(0.45, 0.7), fontsize=12)

fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig2_different_lc.png', dpi=300, bbox_inches='tight')
fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig2_different_lc.pdf', dpi=300, bbox_inches='tight')
plt.show()