'''
Analyze the relationship between NEE seasonl amplitude and regression slopes with atmospheric observations
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

background = '' # '' '_background-ct' '_background-ebg'
seasonal = '' # '' _only_seasonal
metric = 'slope' # slope co2_enhancement_seasonal_amp_ratio co2_enhancement_seasonal_percentile_ratio

if metric == 'slope':
    xlabel = r'Slope of regression with CO$_{2}$ observations'
    xlim = [0, 2]; n_intervals = 5
elif metric == 'co2_enhancement_seasonal_amp_ratio':
    xlabel = r'CO$_{2}$ enhancement seasonal amplitude ratio'
    xlim = [0, 4]; n_intervals = 5
elif metric == 'co2_enhancement_seasonal_percentile_ratio':
    xlabel = r'CO$_{2}$ enhancement seasonal percentile ratio'
    xlim = [0, 4]; n_intervals = 5

# statistical metrics
fitting_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{seasonal}{background}.csv') #_only_seasonal
fitting_df_upscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{seasonal}{background}.csv') #_only_seasonal
fitting_df_inversionsNEE = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{seasonal}{background}.csv') #_only_seasonal
fitting_df = pd.concat([fitting_df_TRENDYv11, fitting_df_upscaledEC, fitting_df_inversionsNEE], ignore_index=True)

# seasonal cycle amplitude
seasonal_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11_ABoVEcore_alllc_unweighted.csv')
seasonal_df_upscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_UpscaledEC_ABoVEcore_alllc_unweighted.csv')
seasonal_df_inversionsNEE = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_inversionsNEE_ABoVEcore_alllc_unweighted.csv')
seasonal_df = pd.concat([seasonal_df_TRENDYv11, seasonal_df_upscaledEC, seasonal_df_inversionsNEE], axis=1)
# seasonal_amp = seasonal_df.max() - seasonal_df.min() # seasonal amplitude for all months
seasonal_amp = seasonal_df.iloc[3:11, :].max() - seasonal_df.iloc[3:11, :].min() # seasonal amplitude for Apr - Nov
# seasonal_amp = seasonal_df.iloc[3:11, :].min() # seasonal maximum carbon uptake for Apr - Nov

# merge the two dataframes
seasonal_amp = seasonal_amp.rename_axis('model_name').reset_index()
fitting_df = pd.merge(fitting_df, seasonal_amp[['model_name', 0]], on='model_name', how='left')
fitting_df = fitting_df.rename(columns={0: 'seasonal_amp'})

# add seasonal amplitude ratio of CO2 enhancement
# option 1: defined by monthly aggregated CO2 enhancement
co2_enhancement_seasonal_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_amplitude_aircraft{seasonal}{background}.csv')
co2_enhancement_seasonal_min_obs = co2_enhancement_seasonal_df['obs'].min()
co2_enhancement_seasonal_max_obs = co2_enhancement_seasonal_df['obs'].max()
co2_enhancement_seasonal_amp_obs = co2_enhancement_seasonal_max_obs - co2_enhancement_seasonal_min_obs
for model_name in fitting_df['model_name']:
    co2_enhancement_seasonal_min = co2_enhancement_seasonal_df[model_name].min()
    co2_enhancement_seasonal_max = co2_enhancement_seasonal_df[model_name].max()
    co2_enhancement_seasonal_amp = co2_enhancement_seasonal_max - co2_enhancement_seasonal_min
    co2_enhancement_seasonal_amp_ratio = co2_enhancement_seasonal_amp / co2_enhancement_seasonal_amp_obs
    fitting_df.loc[fitting_df['model_name'] == model_name, 'co2_enhancement_seasonal_amp_ratio'] = co2_enhancement_seasonal_amp_ratio

# option 2: defined by percentiles of CO2 enhancement
co2_enhancement_percentile_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_percentiles_aircraft{seasonal}{background}.csv')
upper_percentile = 0.95; lower_percentile = 0.05
# upper_percentile = 0.9; lower_percentile = 0.1
# upper_percentile = 0.75; lower_percentile = 0.25
co2_enhancement_seasonal_low_obs = co2_enhancement_percentile_df[co2_enhancement_percentile_df['percentile'] == lower_percentile]['obs'].values[0]
co2_enhancement_seasonal_high_obs = co2_enhancement_percentile_df[co2_enhancement_percentile_df['percentile'] == upper_percentile]['obs'].values[0]
co2_enhancement_seasonal_percentile_obs = co2_enhancement_seasonal_high_obs - co2_enhancement_seasonal_low_obs
for model_name in fitting_df['model_name']:
    co2_enhancement_seasonal_low = co2_enhancement_percentile_df[co2_enhancement_percentile_df['percentile'] == lower_percentile][model_name].values[0]
    co2_enhancement_seasonal_high = co2_enhancement_percentile_df[co2_enhancement_percentile_df['percentile'] == upper_percentile][model_name].values[0]
    co2_enhancement_seasonal_percentile = co2_enhancement_seasonal_high - co2_enhancement_seasonal_low
    co2_enhancement_seasonal_percentile_ratio = co2_enhancement_seasonal_percentile / co2_enhancement_seasonal_percentile_obs
    fitting_df.loc[fitting_df['model_name'] == model_name, 'co2_enhancement_seasonal_percentile_ratio'] = co2_enhancement_seasonal_percentile_ratio


high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #
upscaled_EC = ['X-BASE', 'ABCflux']
inversions = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE']

# set colors
fitting_df.loc[fitting_df['model_name'].isin(high_skill_TRENDY),'color'] = '#396bb8'
fitting_df.loc[fitting_df['model_name'].isin(low_skill_TRENDY),'color'] = '#d4631d'
fitting_df.loc[fitting_df['model_name'].isin (['IBIS']),'color'] = 'grey' ## models with negative Rh
fitting_df.loc[fitting_df['model_name'].isin(upscaled_EC),'color'] = '#56983f'
fitting_df.loc[fitting_df['model_name'].isin(inversions),'color'] = 'black'
# set shapes
fitting_df.loc[fitting_df['model_name'].isin(high_skill_TRENDY + low_skill_TRENDY + ['IBIS']),'shape'] = 'o'
fitting_df.loc[fitting_df['model_name'].isin(upscaled_EC),'shape'] = 'd'
fitting_df.loc[fitting_df['model_name'].isin(inversions),'shape'] = 's'

fig, ax = plt.subplots(figsize=(5,5))
for i in np.arange(fitting_df.shape[0]):
    plt.scatter(fitting_df.loc[i, metric], fitting_df.loc[i, 'seasonal_amp'], marker=fitting_df.loc[i, 'shape'], s=50, facecolor='none', edgecolor=fitting_df.loc[i, 'color']) #,color=fitting_df.loc[i, 'color']

# add a regression line
x = fitting_df[metric]
y = fitting_df['seasonal_amp']
m, b = np.polyfit(x, y, 1)
plt.xlim(xlim)
plt.ylim([0, 5])
x0 = np.arange(xlim[0], xlim[1], 0.1)
plt.plot(x0, m * x0 + b, color='k', linestyle='--', alpha=0.8)
corr, _ = pearsonr(x, y)
plt.text(0.95, 0.05, f'Cor: {"{:.2f}".format(corr)}', fontsize=15, ha='right', va='bottom', transform=ax.transAxes)
plt.xticks(np.linspace(xlim[0], xlim[1], n_intervals), fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(xlabel, fontsize=13)
plt.ylabel('Growing season NEE amplitude '+ '($\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=13)
plt.axvline(x=1, color='gray', linestyle=':', linewidth=1.5)

fig.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/{metric}_NEE_amplitude{background}{seasonal}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/{metric}_NEE_amplitude{background}{seasonal}.pdf', dpi=300, bbox_inches='tight')

plt.show()
