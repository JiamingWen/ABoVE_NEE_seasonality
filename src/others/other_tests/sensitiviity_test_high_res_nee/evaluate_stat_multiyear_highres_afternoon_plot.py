'''plot statistics of model evaluation at various temporal resolutions using afternoon only data'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# whether to keep the monthly result in the plot
replace_monthly = True # True False

# statistics to be plotted
stat_var = 'cor'; xlim = [-0.1, 0.85]; xlabel = r'Correlation with CO$_{2}$ observations'; label = '(a)'
# stat_var = 'slope'; xlim = [-0.2, 2.2]; xlabel = r'Slope of regression with CO$_{2}$ observations'; label = '(b)'
# stat_var = 'intercept'; xlim = [-8, 2]; xlabel = r'Intercept of regression with CO$_{2}$ observations'
# stat_var = 'mean_bias'; xlim = [-8, 2]; xlabel = r'Mean bias compared to CO$_{2}$ observations'; label = '(c)'
# stat_var = 'rmse'; xlim = [0, 15]; xlabel = r'RMSE compared to CO$_{2}$ observations'; label = '(d)'

# read model performance statistics
upscaledEC_df = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC.csv')
inversion_df = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE.csv')

# X-BASE
x_base_df = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_X-BASE_highres.csv')
if replace_monthly:
	x_base_df.loc[x_base_df['model_name'] == 'X-BASE-monthly', x_base_df.columns[1:]] = upscaledEC_df[upscaledEC_df['model_name'] == 'X-BASE'].iloc[:, 1:].values
else:
	x_base_df = pd.concat([x_base_df, upscaledEC_df[upscaledEC_df['model_name'] == 'X-BASE']], ignore_index=True)
	x_base_df['model_name'] = x_base_df['model_name'].replace('X-BASE', 'X-BASE in this study')

# CTE
cte_df = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_CTE_highres.csv')
if replace_monthly:
	cte_df.loc[cte_df['model_name'] == 'CTE-monthly', cte_df.columns[1:]] = inversion_df[inversion_df['model_name'] == 'CTE'].iloc[:, 1:].values
else:
	cte_df = pd.concat([cte_df, inversion_df[inversion_df['model_name'] == 'CTE']], ignore_index=True)
	cte_df['model_name'] = cte_df['model_name'].replace('CTE', 'CTE in this study')

# CT-NOAA
ct_noaa_df = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_CT-NOAA_highres.csv')
if replace_monthly:
	ct_noaa_df.loc[ct_noaa_df['model_name'] == 'CT-NOAA-monthly', ct_noaa_df.columns[1:]] = inversion_df[inversion_df['model_name'] == 'CT-NOAA'].iloc[:, 1:].values
else:
	ct_noaa_df = pd.concat([ct_noaa_df, inversion_df[inversion_df['model_name'] == 'CT-NOAA']], ignore_index=True)
	ct_noaa_df['model_name'] = ct_noaa_df['model_name'].replace('CT-NOAA', 'CT-NOAA in this study')


if replace_monthly:
	fig, ax = plt.subplots(figsize=(7,5))
	markers = ['x', '+']; size = [70, 90]
else:
	fig, ax = plt.subplots(figsize=(7,6))
	markers = ['x', '+', '*']; size = [70, 90, 100]

# X-BASE
plt.scatter(x_base_df[f'{stat_var}'].iloc[-1], x_base_df['model_name'].iloc[-1], marker='d', color='black', facecolor='none', s=70)
for i, marker in enumerate(markers):
	plt.scatter(x_base_df[f'{stat_var}'][len(markers)-1-i], x_base_df['model_name'][len(markers)-1-i], marker=markers[i], color='black', s=size[i])

# CTE
plt.scatter(cte_df[f'{stat_var}'].iloc[-1], cte_df['model_name'].iloc[-1], marker='s', color='black', facecolor='none', s=70)
for i, marker in enumerate(markers):
	plt.scatter(cte_df[f'{stat_var}'][len(markers)-1-i], cte_df['model_name'][len(markers)-1-i], marker=marker, color='black', s=size[i])

# CT-NOAA
plt.scatter(ct_noaa_df[f'{stat_var}'].iloc[-1], ct_noaa_df['model_name'].iloc[-1], marker='s', color='black', facecolor='none', s=70)
for i, marker in enumerate(markers):
	plt.scatter(ct_noaa_df[f'{stat_var}'][len(markers)-1-i], ct_noaa_df['model_name'][len(markers)-1-i], marker=marker, color='black', s=size[i])


'''using afternoon only data'''
# read model performance statistics
upscaledEC_df = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC_afternoon.csv')
inversion_df = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE_afternoon.csv')

# X-BASE
x_base_df = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/afternoon/evaluation_stat_unscaled_X-BASE_highres_afternoon.csv')
if replace_monthly:
	x_base_df.loc[x_base_df['model_name'] == 'X-BASE-monthly', x_base_df.columns[1:]] = upscaledEC_df[upscaledEC_df['model_name'] == 'X-BASE'].iloc[:, 1:].values
else:
	x_base_df = pd.concat([x_base_df, upscaledEC_df[upscaledEC_df['model_name'] == 'X-BASE']], ignore_index=True)
	x_base_df['model_name'] = x_base_df['model_name'].replace('X-BASE', 'X-BASE in this study')

# CTE
cte_df = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/afternoon/evaluation_stat_unscaled_CTE_highres_afternoon.csv')
if replace_monthly:
	cte_df.loc[cte_df['model_name'] == 'CTE-monthly', cte_df.columns[1:]] = inversion_df[inversion_df['model_name'] == 'CTE'].iloc[:, 1:].values
else:
	cte_df = pd.concat([cte_df, inversion_df[inversion_df['model_name'] == 'CTE']], ignore_index=True)
	cte_df['model_name'] = cte_df['model_name'].replace('CTE', 'CTE in this study')

# CT-NOAA
ct_noaa_df = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/afternoon/evaluation_stat_unscaled_CT-NOAA_highres_afternoon.csv')
if replace_monthly:
	ct_noaa_df.loc[ct_noaa_df['model_name'] == 'CT-NOAA-monthly', ct_noaa_df.columns[1:]] = inversion_df[inversion_df['model_name'] == 'CT-NOAA'].iloc[:, 1:].values
else:
	ct_noaa_df = pd.concat([ct_noaa_df, inversion_df[inversion_df['model_name'] == 'CT-NOAA']], ignore_index=True)
	ct_noaa_df['model_name'] = ct_noaa_df['model_name'].replace('CT-NOAA', 'CT-NOAA in this study')


# if replace_monthly:
# 	fig, ax = plt.subplots(figsize=(7,5))
# 	markers = ['x', '+']; size = [70, 90]
# else:
# 	fig, ax = plt.subplots(figsize=(7,6))
# 	markers = ['x', '+', '*']; size = [70, 90, 100]

# X-BASE
plt.scatter(x_base_df[f'{stat_var}'].iloc[-1], x_base_df['model_name'].iloc[-1], marker='d', color='red', facecolor='none', s=70)
for i, marker in enumerate(markers):
	plt.scatter(x_base_df[f'{stat_var}'][len(markers)-1-i], x_base_df['model_name'][len(markers)-1-i], marker=markers[i], color='red', s=size[i])

# CTE
plt.scatter(cte_df[f'{stat_var}'].iloc[-1], cte_df['model_name'].iloc[-1], marker='s', color='red', facecolor='none', s=70)
for i, marker in enumerate(markers):
	plt.scatter(cte_df[f'{stat_var}'][len(markers)-1-i], cte_df['model_name'][len(markers)-1-i], marker=marker, color='red', s=size[i])

# CT-NOAA
plt.scatter(ct_noaa_df[f'{stat_var}'].iloc[-1], ct_noaa_df['model_name'].iloc[-1], marker='s', color='red', facecolor='none', s=70)
for i, marker in enumerate(markers):
	plt.scatter(ct_noaa_df[f'{stat_var}'][len(markers)-1-i], ct_noaa_df['model_name'][len(markers)-1-i], marker=marker, color='red', s=size[i])




'''other settings'''
# section lines
plt.axhline(y = cte_df.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = cte_df.shape[0]+ct_noaa_df.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = cte_df.shape[0]+ct_noaa_df.shape[0]+x_base_df.shape[0]-0.5, color = 'grey', linestyle = '--')

plt.xlim(xlim)
plt.ylim(-0.5, cte_df.shape[0]+ct_noaa_df.shape[0]+x_base_df.shape[0]-0.5)
plt.xlabel(xlabel, fontsize=18)
plt.xticks(fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)

ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left')
fig.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/afternoon/evaluation_stat_high_res_nee_{stat_var}_afternoon.png', dpi=300, bbox_inches='tight')
fig.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/afternoon/evaluation_stat_high_res_nee_{stat_var}_afternoon.pdf', dpi=300, bbox_inches='tight')
plt.show()
