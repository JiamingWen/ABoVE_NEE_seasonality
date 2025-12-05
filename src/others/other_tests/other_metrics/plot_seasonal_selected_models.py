'''plot NEE seasonal cycles for all models and selected models based on the three statistical metrics'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

background = '' # '' '_background-ct' '_background-ebg'
seasonal = '' # '' _only_seasonal

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'

TRENDY_models = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0', 'ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM', 'IBIS'] 
inversion_models = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE'] # excluding models without CARVE coverage
upscaled_EC_models = ['X-BASE', 'ABCflux']

'''stat'''
fitting_df_TRENDYv11_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{seasonal}{background}.csv')
fitting_df_UpscaledEC_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{seasonal}{background}.csv')
fitting_df_inversionsNEE_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{seasonal}{background}.csv')
fitting_df = pd.concat((fitting_df_TRENDYv11_unscaled, fitting_df_UpscaledEC_unscaled, fitting_df_inversionsNEE_unscaled))

# add co2 enhancement range calculated from percentiles
co2_enhancement_percentile_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_percentiles_aircraft{seasonal}{background}.csv')
upper_percentile = 0.95; lower_percentile = 0.05
co2_enhancement_seasonal_low_obs = co2_enhancement_percentile_df[co2_enhancement_percentile_df['percentile'] == lower_percentile]['obs'].values[0]
co2_enhancement_seasonal_high_obs = co2_enhancement_percentile_df[co2_enhancement_percentile_df['percentile'] == upper_percentile]['obs'].values[0]
co2_enhancement_seasonal_percentile_obs = co2_enhancement_seasonal_high_obs - co2_enhancement_seasonal_low_obs
for model_name in fitting_df['model_name']:
    co2_enhancement_seasonal_low = co2_enhancement_percentile_df[co2_enhancement_percentile_df['percentile'] == lower_percentile][model_name].values[0]
    co2_enhancement_seasonal_high = co2_enhancement_percentile_df[co2_enhancement_percentile_df['percentile'] == upper_percentile][model_name].values[0]
    co2_enhancement_seasonal_percentile = co2_enhancement_seasonal_high - co2_enhancement_seasonal_low
    co2_enhancement_seasonal_percentile_ratio = co2_enhancement_seasonal_percentile / co2_enhancement_seasonal_percentile_obs
    fitting_df.loc[fitting_df['model_name'] == model_name, 'co2_enhancement_seasonal_percentile_ratio'] = co2_enhancement_seasonal_percentile_ratio


'''export names of models that rank top 50% for each metric'''
ref_num_dict = {
    'cor': 1,  # correlation
    'slope': 1.0,  # slope
    'intercept': 0.0,  # intercept
    'mean_bias': 0.0,  # mean bias
    'rmse': 0.0,  # RMSE
    'co2_enhancement_seasonal_percentile_ratio': 1.0,  # CO2 enhancement seasonal percentile ratio
}

selected_models = fitting_df['model_name'].unique()
for stat_var in ['cor', 'co2_enhancement_seasonal_percentile_ratio', 'mean_bias']:
    ref_num = ref_num_dict[stat_var]
    fitting_df_cp = fitting_df.copy()
    fitting_df_cp['score'] = np.abs(fitting_df_cp[stat_var] - ref_num)
    top_50pct_n = int(np.ceil(0.5 * len(fitting_df_cp)))
    fitting_df_cp['rank'] = fitting_df_cp['score'].rank(method='first')
    top_models = fitting_df_cp.loc[fitting_df_cp['rank'] <= top_50pct_n, 'model_name'].unique()
    print(f"Top 50% models for {stat_var}: {top_models}")
    selected_models = np.intersect1d(selected_models, top_models)
print(f"Selected models: {selected_models}")


'''plot seasonal cycle'''
seasonal_df_TRENDYv11 = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11_{regionname}_{lcname}_{weightname}.csv")
seasonal_UpscaledEC = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_UpscaledEC_{regionname}_{lcname}_{weightname}.csv")
seasonal_inversionsNEE = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_inversionsNEE_{regionname}_{lcname}_{weightname}.csv")
seasonal_df = pd.concat([seasonal_df_TRENDYv11, seasonal_UpscaledEC, seasonal_inversionsNEE], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Panel 1: All models
axes[0].set_title('(a) All models', fontsize=25)
for model in fitting_df['model_name'].unique():
    if model in seasonal_df_TRENDYv11.columns:
        axes[0].plot(np.arange(4, 12), seasonal_df_TRENDYv11.iloc[3:11][model], label=model, color='#d4631d', linewidth=0.8)
    elif model in seasonal_inversionsNEE.columns:
        axes[0].plot(np.arange(4, 12), seasonal_inversionsNEE.iloc[3:11][model], label=model, color='k', linewidth=0.8)
    elif model in seasonal_UpscaledEC.columns:
        axes[0].plot(np.arange(4, 12), seasonal_UpscaledEC.iloc[3:11][model], label=model, color='#56983F', linewidth=0.8)
axes[0].set_xlabel('Month', fontsize=18)
axes[0].set_ylabel(f'NEE ' + '($\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=18)
axes[0].tick_params(axis='both', labelsize=15)
axes[0].set_xticks(np.arange(4, 12))

# Panel 2: Selected models
axes[1].set_title('(b) Selected models', fontsize=25)
for model in selected_models:
    if model in seasonal_df_TRENDYv11.columns:
        axes[1].plot(np.arange(4, 12), seasonal_df_TRENDYv11.iloc[3:11][model], label=model, color='#d4631d', linewidth=0.8)
    elif model in seasonal_inversionsNEE.columns:
        axes[1].plot(np.arange(4, 12), seasonal_inversionsNEE.iloc[3:11][model], label=model, color='k', linewidth=0.8)
    elif model in seasonal_UpscaledEC.columns:
        axes[1].plot(np.arange(4, 12), seasonal_UpscaledEC.iloc[3:11][model], label=model, color='#56983F', linewidth=0.8)
axes[1].set_xlabel('Month', fontsize=18)
axes[1].tick_params(axis='both', labelsize=15)
axes[1].set_xticks(np.arange(4, 12))

axes[1].text(0.05, 0.05, 'Selected models:\n' + '\n'.join(selected_models), fontsize=14, transform=axes[1].transAxes,
             ha='left', va='bottom', wrap=True)

plt.tight_layout()
plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_cycle_selected_models{background}{seasonal}.png', dpi=300, bbox_inches='tight')
plt.show()


'''report growing season mean and amplitude for all models and selected models'''
seasonal_mean = seasonal_df.iloc[3:11, :].mean()  # seasonal mean for Apr - Nov
seasonal_amp = seasonal_df.iloc[3:11, :].max() - seasonal_df.iloc[3:11, :].min()  # seasonal amplitude for Apr - Nov
# all models
seasonal_mean[fitting_df['model_name']].mean()
seasonal_mean[fitting_df['model_name']].median()
seasonal_mean[fitting_df['model_name']].std()
seasonal_mean[fitting_df['model_name']].max()
seasonal_mean[fitting_df['model_name']].min()
seasonal_mean_range_all = seasonal_mean[fitting_df['model_name']].max() - seasonal_mean[fitting_df['model_name']].min()

seasonal_amp[fitting_df['model_name']].mean()
seasonal_amp[fitting_df['model_name']].median()
seasonal_amp[fitting_df['model_name']].std()
seasonal_amp[fitting_df['model_name']].max()
seasonal_amp[fitting_df['model_name']].min()
seasonal_amp_range_all = seasonal_amp[fitting_df['model_name']].max() - seasonal_amp[fitting_df['model_name']].min()

# selected models
seasonal_mean[selected_models].mean()
seasonal_mean[selected_models].median()
seasonal_mean[selected_models].std()
seasonal_mean[selected_models].max()
seasonal_mean[selected_models].min()
seasonal_mean_range_selected = seasonal_mean[selected_models].max() - seasonal_mean[selected_models].min()

seasonal_amp[selected_models].mean()
seasonal_amp[selected_models].median()
seasonal_amp[selected_models].std()
seasonal_amp[selected_models].max()
seasonal_amp[selected_models].min()
seasonal_amp_range_selected = seasonal_amp[selected_models].max() - seasonal_amp[selected_models].min()

print('Seasonal mean range reduced by', (seasonal_mean_range_all - seasonal_mean_range_selected)/seasonal_mean_range_all * 100, '%')
print('Seasonal amplitude range reduced by', (seasonal_amp_range_all - seasonal_amp_range_selected)/seasonal_amp_range_all * 100, '%')