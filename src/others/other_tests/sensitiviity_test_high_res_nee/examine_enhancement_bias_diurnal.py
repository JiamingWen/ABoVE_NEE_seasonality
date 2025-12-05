'''
Examine if the differences as a result of accounting for NEE diurnal cycle have a relationship with above ground level and other factors
Hypothesis: considering diurnal cycle leads to more CO2 enhancements in the near surface and more CO2 drawdowns in higher altitudes
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
os.chdir('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess


model_types = ['CT-NOAA', 'CTE', 'X-BASE']

# whether to filter observations based on land covers they are most sensitive to
lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname


for model_type in model_types:

    if model_type in ['CT-NOAA', 'CTE']:
        model_names = [model_type+'-3hourly', model_type+'-daily']
    elif model_type == 'X-BASE':
        model_names = [model_type+'-monthly_diurnal', model_type+'-monthly']

    for year in [2012, 2013, 2014, 2017]: #2012, 2013, 2014, 2017

        start_month, end_month, campaign_name = get_campaign_info(year)
        month_num = end_month - start_month + 1

        # read atmospheric observations
        df_airborne = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
        df_influence = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

        # filters for airborne observations
        mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
            (df_influence['ABoVE_influence_fraction'] > 0.5) &
            (df_influence['ocean_influence_fraction'] < 0.3) &
            (df_airborne['CO2_change'] < 30) &
            (df_airborne['CO_change'] < 40))[0]
    
        df_model_diurnal_year = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_{model_names[0]}.csv')
        df_model_nodiurnal_year = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_{model_names[1]}.csv')

        df_model_diurnal_year = df_model_diurnal_year.rename(columns={f"{model_type}": f"{model_type}_diurnal"})
        df_model_nodiurnal_year = df_model_nodiurnal_year.rename(columns={f"{model_type}": f"{model_type}_nodiurnal"})
        df_year = pd.concat((df_airborne.loc[mask_id],
                                 df_model_diurnal_year[f"{model_type}_diurnal"].loc[mask_id],
                                 df_model_nodiurnal_year[f"{model_type}_nodiurnal"].loc[mask_id]), axis=1)

        if year == 2012:
            df = df_year
        else:
            df = pd.concat((df, df_year), axis=0)
    
    df = df.reset_index(drop=True)
    df['diurnal_diff'] = df[f'{model_type}_diurnal'] - df[f'{model_type}_nodiurnal']
    df['local_hour'] = [datetime.strptime(df['footprint_time_AKT'][index], "%Y-%m-%d %H:%M:%S-0%f:00").hour for index, _ in df.iterrows()]
    df['local_month'] = [datetime.strptime(df['footprint_time_AKT'][index], "%Y-%m-%d %H:%M:%S-0%f:00").month for index, _ in df.iterrows()]


    '''plot a histogram'''
    plt.figure(figsize=(5,3))
    plt.hist(df['diurnal_diff'], bins=np.arange(-60, 60, 2), color='gray', edgecolor='black')
    plt.xlabel('Diurnal - No Diurnal (ppm)', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.title(f'{model_type}', fontsize=15)
    plt.xlim([-40, 40])
    plt.ylim([0, 7500])
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.show()


    '''plot diurnal difference vs. above ground height'''
    plt.figure(figsize=(5,3))
    # plt.scatter(df['footprint_agl'], df['diurnal_diff'], s=5, alpha=0.5)
    plt.hexbin(df['footprint_agl'], df['diurnal_diff'], gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(label='Point Density')
    plt.xlabel('Above Ground Level (m)', fontsize=13)
    plt.ylabel('Diurnal - No Diurnal (ppm)', fontsize=13)
    plt.ylim([-60, 60])
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.title(f'{model_type}', fontsize=15)
    plt.show()

    # plot a non-parametric fitting curve using LOWESS
    plt.figure(figsize=(5,3))
    lowess_smoothed = lowess(df['diurnal_diff'], df['footprint_agl'], frac=0.3)
    plt.plot(lowess_smoothed[:, 0], lowess_smoothed[:, 1], color='red', label='LOWESS Fit')
    plt.xlabel('Above Ground Level (m)', fontsize=13)
    plt.ylabel('Diurnal - No Diurnal (ppm)', fontsize=13)
    plt.ylim([-2, 2])
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.title(f'{model_type}', fontsize=15)
    plt.show()


    '''plot diurnal difference vs. local hour'''
    plt.figure(figsize=(7,3))
    df.boxplot(column='diurnal_diff', by='local_hour', grid=False, showfliers=False, widths=0.6)
    plt.suptitle('') # to remove the automatic title
    plt.xlabel('Local Hour (AKT)', fontsize=13)
    plt.ylabel('Diurnal - No Diurnal (ppm)', fontsize=13)
    plt.ylim([-10, 10])
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.title(f'{model_type}', fontsize=15)
    plt.show() 


    '''plot diurnal difference vs. month'''
    plt.figure(figsize=(4,3))
    df.boxplot(column='diurnal_diff', by='local_month', grid=False, showfliers=False, widths=0.6)
    plt.suptitle('') # to remove the automatic title
    plt.xlabel('Month', fontsize=13)
    plt.ylabel('Diurnal - No Diurnal (ppm)', fontsize=13)
    plt.xticks(fontsize=12)
    plt.ylim([-10, 10])
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.title(f'{model_type}', fontsize=15)
    plt.show() 


    '''plot diurnal difference vs. local hour x month'''
    heatmap_data = df.pivot_table(index='local_hour', columns='local_month', values='diurnal_diff', aggfunc='mean')
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm', origin='lower', vmin=-10, vmax=10)
    plt.colorbar(label='Mean Diurnal - No Diurnal (ppm)')
    plt.xticks(ticks=np.arange(heatmap_data.columns.size), labels=heatmap_data.columns, fontsize=12)
    plt.yticks(ticks=np.arange(heatmap_data.index.size), labels=heatmap_data.index, fontsize=12)
    plt.xlabel('Month', fontsize=13)
    plt.ylabel('Local Hour (AKT)', fontsize=13)
    plt.title(f'{model_type}', fontsize=15)
    plt.show()


    '''plot diurnal difference vs. agl x month'''
    df['footprint_agl_bin'] = pd.cut(df['footprint_agl'], bins=np.arange(0, df['footprint_agl'].max() + 100, 100))
    heatmap_data = df.pivot_table(index='footprint_agl_bin', columns='local_month', values='diurnal_diff', aggfunc='mean')
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm', origin='lower', vmin=-10, vmax=10)
    plt.colorbar(label='Mean Diurnal - No Diurnal (ppm)')
    plt.xticks(ticks=np.arange(heatmap_data.columns.size), labels=heatmap_data.columns, fontsize=12)
    plt.yticks(ticks=np.arange(heatmap_data.index.size), labels=heatmap_data.index, fontsize=12)
    plt.xlabel('Month', fontsize=13)
    plt.ylabel('Above Ground Level (m)', fontsize=13)
    plt.title(f'{model_type}', fontsize=15)
    plt.show()