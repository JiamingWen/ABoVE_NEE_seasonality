'''plot seasonal cycle of scaled remote sensing variables'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLSResults

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

weightname = 'unweighted' #unweighted
regionname = 'ABoVEcore'

# standardize with minumum NEE
def scale_minumum (vec):
    return -vec / np.min(vec)

scale_fun = scale_minumum
ylim = [-1.2,1.2]

seasonal_reference = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_reference_{regionname}_{lcname}_{weightname}.csv')
fitting_df_reference_only_seasonal = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_scaled_reference{lc_filestr}_only_seasonal.csv')
reference_names = ['APAR', 'GOME2_SIF'] # 'PAR', 'FPAR', 
titlename = ['APAR', 'SIF'] #'PAR', 'FPAR', 

# reference
seasonal_df_inversions = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_inversionsNEE_{regionname}_{lcname}_{weightname}.csv")
inversion_names = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE'] # excluding models without CARVE coverage
seasonal_df_subset_inversion = seasonal_df_inversions[inversion_names]
seasonal_df_subset_inversion = seasonal_df_subset_inversion.apply(scale_fun, axis=0)
seasonal_mean_ensemble_inversion = seasonal_df_subset_inversion.mean(axis=1)
seasonal_std_ensemble_inversion = seasonal_df_subset_inversion.std(axis=1)

# fig, ax = plt.subplots(1, 4, figsize=(17, 3.2))
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
subplot_id = 0

for i in np.arange(len(reference_names)):

    seasonal_vec = seasonal_reference[reference_names[i]]
    results1 = OLSResults.load(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/regression/reference_only_seasonal_{reference_names[i]}{lc_filestr}.pickle")
    fitted_slope = results1.params[0]
    fitted_intercept = results1.params[1]
    seasonal_vec_scaled = seasonal_vec * fitted_slope + fitted_intercept

    seasonal_vec_scaled = scale_fun(seasonal_vec_scaled)

    subplot_id += 1
    ax1 = plt.subplot(1, 2, subplot_id)

    # inversions
    plt.fill_between(np.arange(1,13), seasonal_mean_ensemble_inversion-seasonal_std_ensemble_inversion, seasonal_mean_ensemble_inversion+seasonal_std_ensemble_inversion, alpha=0.2,color='black') #,edgecolor='none'
    # scaled reference
    plt.plot(np.arange(1,13),seasonal_vec_scaled, linestyle='-',color='purple', linewidth=2)
    mean_seasonal_NEE_diff = np.mean(abs(seasonal_mean_ensemble_inversion[3:11]-seasonal_vec_scaled[3:11])) # only growing season
    cor = fitting_df_reference_only_seasonal.loc[fitting_df_reference_only_seasonal['model_name'] == reference_names[i], 'cor'].values[0]
    subtitle = chr(ord('`')+i+1)
    plt.annotate(f'({subtitle}) {titlename[i]}', (4.5, ylim[1]-(ylim[1]-ylim[0])/8), fontsize=18)
    # plt.annotate(f'Cor={"{:.2f}".format(cor)}', (8, ylim[0]+(ylim[1]-ylim[0])/6), fontsize=14)
    # plt.annotate(f'MAD={"{:.2f}".format(mean_seasonal_NEE_diff)}', (8, ylim[0]+(ylim[1]-ylim[0])/14), fontsize=14)

    plt.xlim(4,11) #4,11
    plt.ylim(ylim)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax1.set_xticks(np.arange(4,12))
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Rescaled NEE', fontsize=16)

plt.subplots_adjust(wspace=0.4, hspace=0.35)

plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/seasonal_scaled_remote_sensing.png', dpi=300, bbox_inches='tight')
plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/seasonal_scaled_remote_sensing.pdf', dpi=300, bbox_inches='tight')
plt.show()
