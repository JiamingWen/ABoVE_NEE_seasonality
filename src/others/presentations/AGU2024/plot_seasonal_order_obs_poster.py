import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLSResults

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

# remember to check the high- and low- groups
high_model_subset = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0'] # , 'IBIS' #'ISBA-CTRIP', 'LPJ', 'LPX-Bern', 'SDGVM', 'CLASSIC', 'CLM5.0'
low_model_subset = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #, 'VISIT-NIES'   

weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'

# standardize with maximum
def scale_maximum (vec):
    return vec / np.max(abs(vec))

# standardize with minumum NEE
def scale_minumum (vec):
    return -vec / np.min(vec)

# standardize with maximum and minimum
def scale_maximum_minimum (vec):
    return (vec - np.min(vec)) / (np.max(vec) - np.min(vec))

# stat
fitting_df_TRENDYv11_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_only_seasonal.csv')
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted = fitting_df_TRENDYv11_unscaled_only_seasonal.sort_values('cor', ascending=False)
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(high_model_subset),'color'] = '#5986cb'
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(low_model_subset),'color'] = '#e57f3f'
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(['IBIS']),'color'] = '#5986cb' #'grey'


##########################################
# NEE
scale_fun = scale_minumum
ylim = [-1.2,1.2]

seasonal_df_TRENDYv11NEE = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_TRENDYv11_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11NEE = seasonal_df_TRENDYv11NEE.apply(scale_fun, axis=0)

seasonal_df_inversions = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_inversionsNEE_{regionname}_{lcname}_{weightname}.csv")
inversion_names = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE'] # excluding models without CARVE coverage
seasonal_df_subset_inversion = seasonal_df_inversions[inversion_names]
seasonal_df_subset_inversion = seasonal_df_subset_inversion.apply(scale_fun, axis=0)
seasonal_mean_ensemble_inversion = seasonal_df_subset_inversion.mean(axis=1)
seasonal_std_ensemble_inversion = seasonal_df_subset_inversion.std(axis=1)

# fig, ax = plt.subplots(16, 1, figsize=(4, 60))
fig, ax = plt.subplots(8, 2, figsize=(8, 30))
subplot_id = 0

mean_seasonal_NEE_diff_df = pd.DataFrame()
for i in np.arange(fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.shape[0]):

    model_name = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.iloc[i]['model_name']
    color = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.iloc[i]['color']

    subplot_id += 1
    # ax1 = plt.subplot(16, 1, subplot_id)
    ax1 = plt.subplot(8, 2, subplot_id)

    # inversions
    plt.fill_between(np.arange(1,13), seasonal_mean_ensemble_inversion-seasonal_std_ensemble_inversion, seasonal_mean_ensemble_inversion+seasonal_std_ensemble_inversion, alpha=0.2,color='black') #,edgecolor='none'

    # individual TRENDY models
    plt.plot(np.arange(1,13),seasonal_df_TRENDYv11NEE[model_name], linestyle='-',color=color, linewidth=2)
    
    # mean_seasonal_NEE_diff = np.mean(abs(seasonal_mean_ensemble_inversion-seasonal_df_TRENDYv11NEE[model_name])) # annual
    mean_seasonal_NEE_diff = np.mean(abs(seasonal_mean_ensemble_inversion[3:11]-seasonal_df_TRENDYv11NEE[model_name][3:11])) # only growing season
    cor = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.iloc[i]['cor']
    subtitle = chr(ord('`')+i+1)
    plt.annotate(f'({subtitle}) {model_name}', (4.5, ylim[1]-(ylim[1]-ylim[0])/8), fontsize=18)
    plt.annotate(f'Cor={"{:.2f}".format(cor)}', (8, ylim[0]+(ylim[1]-ylim[0])/6), fontsize=14)
    plt.annotate(f'MAD={"{:.2f}".format(mean_seasonal_NEE_diff)}', (8, ylim[0]+(ylim[1]-ylim[0])/14), fontsize=14)

    plt.xlim(4,11) #4,11
    plt.ylim(ylim)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax1.set_xticks(np.arange(4,12))
    plt.xlabel('Month', fontsize=16)
    plt.ylabel('Standardized NEE', fontsize=16)

    mean_seasonal_NEE_diff_df = pd.concat((mean_seasonal_NEE_diff_df, pd.DataFrame([[model_name, cor, mean_seasonal_NEE_diff]], columns=['model_name', 'cor', 'mean_seasonal_diff'])))

plt.subplots_adjust(wspace=0.4, hspace=0.35)
plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/figures/seasonal_order_obs_{regionname}_{lcname}_{weightname}_poster.png', dpi=300, bbox_inches='tight')
plt.show()

########################################################
mean_seasonal_NEE_diff_df_reset = mean_seasonal_NEE_diff_df.reset_index()
fig, ax = plt.subplots(figsize=(4,4))
for i in np.arange(mean_seasonal_NEE_diff_df_reset.shape[0]):
    plt.scatter(mean_seasonal_NEE_diff_df_reset.loc[i, 'cor'], mean_seasonal_NEE_diff_df_reset.loc[i, 'mean_seasonal_diff'],color=fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.iloc[i]['color'], s=50)
# add a regression line
df_for_reg = mean_seasonal_NEE_diff_df_reset
x = df_for_reg['cor']
y = df_for_reg['mean_seasonal_diff']
m, b = np.polyfit(x, y, 1)
x0 = np.arange(-0.1,0.9,0.1)
plt.plot(x0, m * x0 + b, color='k', linestyle='--', alpha=0.8)
corr, _ = pearsonr(x, y)
plt.text(0.5, 0.55, f'Cor: {"{:.2f}".format(corr)}', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'Correlation with CO$_{2}$ observations', fontsize=13)
plt.ylabel('MAD of standardized seasonal cycle', fontsize=13)
plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/figures/cor_MAD_poster.png', dpi=300, bbox_inches='tight')
plt.show()