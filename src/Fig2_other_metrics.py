'''
plot summary figure for model performance, e.g., mean bias, regression slopes/intercepts, RMSE
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

stat_var = 'cor'; xlim = [-0.2, 0.85]; xlabel = r'Correlation with CO$_{2}$ observations'
# stat_var = 'slope'; xlim = [-0.2, 2.2]; xlabel = r'Slope of regression with CO$_{2}$ observations'
# stat_var = 'intercept'; xlim = [-8, 2]; xlabel = r'Intercept of regression with CO$_{2}$ observations'
# stat_var = 'mean_bias'; xlim = [-8, 2]; xlabel = r'Mean bias compared to CO$_{2}$ observations'
# stat_var = 'rmse'; xlim = [0, 15]; xlabel = r'RMSE compared to CO$_{2}$ observations'

# unscaled variables (without linear regression)
fitting_df_TRENDYv11_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}.csv')
# fitting_df_TRENDYv11_unscaled = fitting_df_TRENDYv11_unscaled.loc[~fitting_df_TRENDYv11_unscaled['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}.csv')
fitting_df_inversions_unscaled = fitting_df_inversions_unscaled.loc[~fitting_df_inversions_unscaled['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_unscaled.loc[fitting_df_inversions_unscaled['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_UpscaledEC_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}.csv')

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
plt.ylim(-1, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]-0.5) #+fitting_df_regression_all_sorted.shape[0]
plt.xlabel(xlabel, fontsize=18)
plt.xticks(fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)


colors = fitting_df_TRENDYv11_sorted['color'].values.tolist() + ['#56983f']*fitting_df_UpscaledEC_sorted.shape[0] + ['black']*fitting_df_inversions_sorted.shape[0]
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)

ax.annotate("Atmospheric Inversions", (xlim[0]+(xlim[1]-xlim[0])/20, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-1.5), fontsize=15)
ax.annotate("Upscaled EC", (xlim[0]+(xlim[1]-xlim[0])/20, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_UpscaledEC_sorted.shape[0]-1.3), fontsize=15)
ax.annotate("TRENDY TBMs", (xlim[0]+(xlim[1]-xlim[0])/20, fitting_df_TRENDYv11_sorted.shape[0]-2.5), fontsize=15)


##########################################################################################
# add additional evaluation with mean seasonal cycle
fitting_df_TRENDYv11_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_only_seasonal.csv')
# fitting_df_TRENDYv11_unscaled_only_seasonal = fitting_df_TRENDYv11_unscaled_only_seasonal.loc[~fitting_df_TRENDYv11_unscaled_only_seasonal['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_only_seasonal.csv')
fitting_df_inversions_unscaled_only_seasonal = fitting_df_inversions_unscaled_only_seasonal.loc[~fitting_df_inversions_unscaled_only_seasonal['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions_unscaled_only_seasonal.loc[fitting_df_inversions_unscaled_only_seasonal['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_UpscaledEC_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_only_seasonal.csv')

fitting_df_TRENDYv11_merge = pd.merge(fitting_df_TRENDYv11_sorted, fitting_df_TRENDYv11_unscaled_only_seasonal, on='model_name', how='outer', suffixes=('_original', '_only_seasonal'))
fitting_df_inversions_merge = pd.merge(fitting_df_inversions_sorted, fitting_df_inversions_unscaled_only_seasonal, on='model_name', how='outer', suffixes=('_original', '_only_seasonal'))
fitting_df_UpscaledEC_merge = pd.merge(fitting_df_UpscaledEC_sorted, fitting_df_UpscaledEC_unscaled_only_seasonal, on='model_name', how='outer', suffixes=('_original', '_only_seasonal'))

p1b = plt.scatter(fitting_df_TRENDYv11_merge[f'{stat_var}_only_seasonal'], fitting_df_TRENDYv11_merge['model_name'], marker='o', color=fitting_df_TRENDYv11_sorted['color'], s=70) #, alpha=0.8
p2b = plt.scatter(fitting_df_UpscaledEC_merge[f'{stat_var}_only_seasonal'], fitting_df_UpscaledEC_merge['model_name'], marker='d', color='#56983f', s=80)
p3b = plt.scatter(fitting_df_inversions_merge[f'{stat_var}_only_seasonal'], fitting_df_inversions_merge['model_name'], marker='s', color='black', s=60) #, label='Mean seasonal cycle'

plt.legend([(p1a, p2a, p3a), (p1b, p2b, p3b)], ['Original data', 'Mean seasonal cycle'],
           loc='best', fontsize=14, handletextpad=1.5, scatterpoints=1, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None, pad=2)}, frameon=True, borderpad=1, labelspacing=0.5)

fig.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig2_other_metrics_{stat_var}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig2_other_metrics_{stat_var}.pdf', dpi=300, bbox_inches='tight')
plt.show()


'''export names of models that rank top 50% for each metric'''
fitting_df = pd.concat((fitting_df_TRENDYv11_unscaled, fitting_df_UpscaledEC_unscaled, fitting_df_inversions_unscaled), axis=0)
ref_num_dict = {
    'cor': 1,  # correlation
    'slope': 1.0,  # slope
    'intercept': 0.0,  # intercept
    'mean_bias': 0.0,  # mean bias
    'rmse': 0.0,  # RMSE
}
ref_num = ref_num_dict[stat_var]
fitting_df['score'] = np.abs(fitting_df[stat_var] - ref_num)
top_50pct_n = int(np.ceil(0.5 * len(fitting_df)))
fitting_df['rank'] = fitting_df['score'].rank(method='first')
top_models = fitting_df.loc[fitting_df['rank'] <= top_50pct_n, 'model_name'].unique()
print(top_models)
