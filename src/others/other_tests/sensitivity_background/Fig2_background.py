'''
plot summary figure for model performance (correlation with observed CO2 enhancement)
evaluation with different background calculations
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
fitting_df_TRENDYv11_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}.csv')
# fitting_df_TRENDYv11_unscaled = fitting_df_TRENDYv11_unscaled.loc[~fitting_df_TRENDYv11_unscaled['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}.csv')
fitting_df_inversions_unscaled = fitting_df_inversions_unscaled.loc[~fitting_df_inversions_unscaled['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_unscaled.loc[fitting_df_inversions_unscaled['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_UpscaledEC_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}.csv')

# scaled variables (with linear regression)
fitting_df_reference_scaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_scaled_reference{lc_filestr}.csv')
fitting_df_reference_scaled = fitting_df_reference_scaled.loc[fitting_df_reference_scaled['model_name'].isin(['APAR', 'GOME2_SIF']), :]  #'APAR', 'FPAR', 'LAI', 'PAR'
fitting_df_reference_scaled.loc[fitting_df_reference_scaled['model_name'] == 'GOME2_SIF','model_name'] = 'SIF'

# sort for each category
fitting_df_TRENDYv11_sorted = fitting_df_TRENDYv11_unscaled.sort_values(f'{stat_var}')
fitting_df_inversions_sorted = fitting_df_inversions_unscaled.sort_values(f'{stat_var}')
fitting_df_UpscaledEC_sorted = fitting_df_UpscaledEC_unscaled.sort_values(f'{stat_var}')
fitting_df_reference_sorted = fitting_df_reference_scaled.sort_values(f'{stat_var}')

# set colors
high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(high_skill_TRENDY),'color'] = '#396bb8'
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(low_skill_TRENDY),'color'] = '#d4631d'
fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin (['IBIS']),'color'] = 'grey' ## models with negative Rh


fig, ax = plt.subplots(figsize=(7,10))

labelname = f"Airborne Profiles"

plt.scatter(fitting_df_TRENDYv11_sorted[f'{stat_var}'], fitting_df_TRENDYv11_sorted['model_name'], marker='o', color='black', label=labelname, s=70, facecolors='none', linewidths=2) #, color=fitting_df_TRENDYv11_sorted['color']
plt.scatter(fitting_df_UpscaledEC_sorted[f'{stat_var}'], fitting_df_UpscaledEC_sorted['model_name'], marker='o', color='black', s=70, facecolors='none', linewidths=2) #, color='#6db753'
plt.scatter(fitting_df_inversions_sorted[f'{stat_var}'], fitting_df_inversions_sorted['model_name'], marker='o', color='black', s=70, facecolors='none', linewidths=2)  #, color='black'
plt.scatter(fitting_df_reference_sorted[f'{stat_var}'], fitting_df_reference_sorted['model_name'], marker='o', color='black', s=70, facecolors='none', linewidths=2)  #, color='purple'

plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')

plt.xlim(xlim)
plt.ylim(-1, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_reference_sorted.shape[0]-0.5)
plt.xlabel(r'Correlation with CO$_{2}$ observations', fontsize=18)
plt.xticks(ticks=np.arange(-0.2, 0.9, 0.1), fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)


colors = fitting_df_TRENDYv11_sorted['color'].values.tolist() + ['#6db753']*fitting_df_UpscaledEC_sorted.shape[0] + ['black']*fitting_df_inversions_sorted.shape[0] + ['purple']*fitting_df_reference_sorted.shape[0]
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)

plt.annotate("Remote Sensing", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_reference_sorted.shape[0]-1.5), fontsize=15)
plt.annotate("Atmospheric Inversions", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-1.5), fontsize=15)
plt.annotate("Upscaled EC", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]-1.3), fontsize=15)
plt.annotate("TRENDY", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]-1.5), fontsize=15)


##############################################################
# overlay with results for CT backgrounds

# unscaled variables (without linear regression)
fitting_df_TRENDYv11_unscaled_background_ct = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_background-ct.csv')
# fitting_df_TRENDYv11_unscaled_background_ct = fitting_df_TRENDYv11_unscaled_background_ct.loc[~fitting_df_TRENDYv11_unscaled_background_ct['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled_background_ct = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_background-ct.csv')
fitting_df_inversions_unscaled_background_ct = fitting_df_inversions_unscaled_background_ct.loc[~fitting_df_inversions_unscaled_background_ct['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_unscaled_background_ct.loc[fitting_df_inversions_unscaled_background_ct['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_UpscaledEC_unscaled_background_ct = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_background-ct.csv')

# scaled variables (with linear regression)
fitting_df_reference_scaled_background_ct = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_scaled_reference{lc_filestr}_background-ct.csv')
fitting_df_reference_scaled_background_ct = fitting_df_reference_scaled_background_ct.loc[fitting_df_reference_scaled_background_ct['model_name'].isin(['APAR', 'GOME2_SIF']), :]  #'APAR', 'FPAR', 'LAI', 'PAR'
fitting_df_reference_scaled_background_ct.loc[fitting_df_reference_scaled_background_ct['model_name'] == 'GOME2_SIF','model_name'] = 'SIF'

fitting_df_TRENDYv11_merge_ct = pd.merge(fitting_df_TRENDYv11_sorted, fitting_df_TRENDYv11_unscaled_background_ct, on='model_name', how='outer', suffixes=('_airborne_profiles', '_background_ct'))
fitting_df_inversions_merge_ct = pd.merge(fitting_df_inversions_sorted, fitting_df_inversions_unscaled_background_ct, on='model_name', how='outer', suffixes=('_airborne_profiles', '_background_ct'))
fitting_df_UpscaledEC_merge_ct = pd.merge(fitting_df_UpscaledEC_sorted, fitting_df_UpscaledEC_unscaled_background_ct, on='model_name', how='outer', suffixes=('_airborne_profiles', '_background_ct'))
fitting_df_reference_merge_ct = pd.merge(fitting_df_reference_scaled, fitting_df_reference_scaled_background_ct, on='model_name', how='outer', suffixes=('_airborne_profiles', '_background_ct'))

color = 'blue'
labelname = f"Carbon Tracker"

plt.scatter(fitting_df_TRENDYv11_merge_ct[f'{stat_var}_background_ct'], fitting_df_TRENDYv11_merge_ct['model_name'], marker='o', color=color, facecolors='none', label=labelname, s=70)
plt.scatter(fitting_df_inversions_merge_ct[f'{stat_var}_background_ct'], fitting_df_inversions_merge_ct['model_name'], marker='o', color=color, facecolors='none', s=70)
plt.scatter(fitting_df_UpscaledEC_merge_ct[f'{stat_var}_background_ct'], fitting_df_UpscaledEC_merge_ct['model_name'], marker='o', color=color, facecolors='none', s=70)
plt.scatter(fitting_df_reference_merge_ct[f'{stat_var}_background_ct'], fitting_df_reference_merge_ct['model_name'], marker='o', color=color, facecolors='none', s=70)


##############################################################
# overlay with results for empirical backgrounds

# unscaled variables (without linear regression)
fitting_df_TRENDYv11_unscaled_background_ebg = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_background-ebg.csv')
# fitting_df_TRENDYv11_unscaled_background_ebg = fitting_df_TRENDYv11_unscaled_background_ebg.loc[~fitting_df_TRENDYv11_unscaled_background_ebg['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled_background_ebg = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_background-ebg.csv')
fitting_df_inversions_unscaled_background_ebg = fitting_df_inversions_unscaled_background_ebg.loc[~fitting_df_inversions_unscaled_background_ebg['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_unscaled_background_ebg.loc[fitting_df_inversions_unscaled_background_ebg['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_UpscaledEC_unscaled_background_ebg = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_background-ebg.csv')

# scaled variables (with linear regression)
fitting_df_reference_scaled_background_ebg = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_scaled_reference{lc_filestr}_background-ebg.csv')
fitting_df_reference_scaled_background_ebg = fitting_df_reference_scaled_background_ebg.loc[fitting_df_reference_scaled_background_ebg['model_name'].isin(['APAR', 'GOME2_SIF']), :]  #'APAR', 'FPAR', 'LAI', 'PAR'
fitting_df_reference_scaled_background_ebg.loc[fitting_df_reference_scaled_background_ebg['model_name'] == 'GOME2_SIF','model_name'] = 'SIF'

fitting_df_TRENDYv11_merge_ebg = pd.merge(fitting_df_TRENDYv11_sorted, fitting_df_TRENDYv11_unscaled_background_ebg, on='model_name', how='outer', suffixes=('_airborne_profiles', '_background_ebg'))
fitting_df_inversions_merge_ebg = pd.merge(fitting_df_inversions_sorted, fitting_df_inversions_unscaled_background_ebg, on='model_name', how='outer', suffixes=('_airborne_profiles', '_background_ebg'))
fitting_df_UpscaledEC_merge_ebg = pd.merge(fitting_df_UpscaledEC_sorted, fitting_df_UpscaledEC_unscaled_background_ebg, on='model_name', how='outer', suffixes=('_airborne_profiles', '_background_ebg'))
fitting_df_reference_merge_ebg = pd.merge(fitting_df_reference_scaled, fitting_df_reference_scaled_background_ebg, on='model_name', how='outer', suffixes=('_airborne_profiles', '_background_ebg'))

color = 'red'
labelname = f"Empirical Background"

plt.scatter(fitting_df_TRENDYv11_merge_ebg[f'{stat_var}_background_ebg'], fitting_df_TRENDYv11_merge_ebg['model_name'], marker='o', color=color, facecolors='none', label=labelname, s=70)
plt.scatter(fitting_df_inversions_merge_ebg[f'{stat_var}_background_ebg'], fitting_df_inversions_merge_ebg['model_name'], marker='o', color=color, facecolors='none', s=70)
plt.scatter(fitting_df_UpscaledEC_merge_ebg[f'{stat_var}_background_ebg'], fitting_df_UpscaledEC_merge_ebg['model_name'], marker='o', color=color, facecolors='none', s=70)
plt.scatter(fitting_df_reference_merge_ebg[f'{stat_var}_background_ebg'], fitting_df_reference_merge_ebg['model_name'], marker='o', color=color, facecolors='none', s=70)


plt.legend(bbox_to_anchor=(0.1, 0.7), fontsize=14)

fig.savefig('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_background/Fig2_background.png', dpi=300, bbox_inches='tight')
fig.savefig('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_background/Fig2_background.pdf', dpi=300, bbox_inches='tight')
plt.show()