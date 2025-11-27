'''decompose the variance of modeled CO2 enhancement due to NEE seasonal and diurnal cycles'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from datetime import datetime
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info

model_type = 'X-BASE'

fig, axes = plt.subplots(3, 5, sharex=False, sharey=True, figsize=(14, 6))
for year_id, year in enumerate([2012, 2013, 2014, 2017]): 

    print(f"Processing year {year}...")

    start_month, end_month, campaign_name = get_campaign_info(year)
    month_num = end_month - start_month + 1

    # read atmospheric observations
    df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
    df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

    # filters for airborne observations
    mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
        (df_influence['ABoVE_influence_fraction'] > 0.5) &
        (df_influence['ocean_influence_fraction'] < 0.3) &
        (df_airborne['CO2_change'] < 30) &
        (df_airborne['CO_change'] < 40))[0]

    df_airborne['co2_total'] = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_X-BASE-monthly_diurnal.csv')[f"{model_type}"]
    df_airborne['co2_seasonal'] = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_X-BASE-monthly.csv')[f"{model_type}"]
    df_airborne['co2_diurnal'] = df_airborne['co2_total'] - df_airborne['co2_seasonal']
    
    df_year = df_airborne.loc[mask_id]

    '''calculate correlation and variance'''
    print(pearsonr(df_year['co2_total'], df_year['co2_seasonal']))
    print(pearsonr(df_year['co2_total'], df_year['co2_diurnal']))
    print(pearsonr(df_year['co2_seasonal'], df_year['co2_diurnal']))

    print(np.std(df_year['co2_total']))
    print(np.std(df_year['co2_seasonal']))
    print(np.std(df_year['co2_diurnal']))

    if year == 2012:
        df_all_years = df_year
    else:
        df_all_years = pd.concat([df_all_years, df_year], axis=0)

    df_year['footprint_time_AKT'] = pd.to_datetime(df_year['footprint_time_AKT'], utc=False)
    
    axes[0,year_id].scatter(df_year['footprint_time_AKT'], df_year['co2_total'], edgecolor='black', facecolors='black', s=10, alpha=0.05)
    axes[1,year_id].scatter(df_year['footprint_time_AKT'], df_year['co2_seasonal'], edgecolor='red', facecolors='red', s=10, alpha=0.05)
    axes[2,year_id].scatter(df_year['footprint_time_AKT'], df_year['co2_diurnal'], edgecolor='blue', facecolors='blue', s=10, alpha=0.05)
    
    # axis
    for r in range(3):
        months = range(4, 12)
        tick_dates = [pd.Timestamp(year=year, month=m, day=1) for m in months]
        axes[r,year_id].set_xlim(pd.Timestamp(year=year, month=4, day=1),
                                pd.Timestamp(year=year, month=11, day=30))
        axes[r,year_id].set_xticks(tick_dates)
        axes[r, year_id].set_xticklabels([str(m) for m in months], fontsize=12)
        axes[r,year_id].axhline(0, color='gray', linestyle='--', linewidth=1)
        axes[r,year_id].set_ylim(-30, 30)

    # set column title for each year (top row only)
    axes[0,year_id].set_title(str(year), fontsize=16)

    if year_id == 0:
        axes[0,year_id].text(0.02, 0.85, '(a) Seasonal + diurnal', transform=axes[0,year_id].transAxes, fontsize=13)
        axes[1,year_id].text(0.02, 0.85, '(b) Seasonal only', transform=axes[1,year_id].transAxes, fontsize=13)
        axes[2,year_id].text(0.02, 0.85, '(c) Diurnal only', transform=axes[2,year_id].transAxes, fontsize=13)


fig.text(0.45, 0, 'Month', ha='center', va='center', fontsize=16)
fig.supylabel('CO$_2$ enhancements (ppm)', fontsize=16)

# add histogram column on the far right
bins = np.arange(-30, 31, 2)

# total
axes[0, 4].hist(df_all_years['co2_total'], bins=bins, orientation='horizontal',
                color='black', alpha=0.6, edgecolor='black')

# seasonal
axes[1, 4].hist(df_all_years['co2_seasonal'], bins=bins, orientation='horizontal',
                color='red', alpha=0.6, edgecolor='red')

# diurnal
axes[2, 4].hist(df_all_years['co2_diurnal'], bins=bins, orientation='horizontal',
                color='blue', alpha=0.6, edgecolor='blue')
fig.text(0.9, 0, 'Counts', ha='center', va='center', fontsize=16)

for r in range(3):
    axes[r, 4].tick_params(axis='y', labelleft=False)
    axes[r, 4].set_xlim(0, 8000)
    axes[r, 4].set_ylim(-30, 30)
    axes[r, 4].tick_params(axis='x', labelsize=12)

plt.tight_layout()
plt.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src/others/other_tests/sensitiviity_test_high_res_nee/decompose_var_monthly_diurnal.png', dpi=300)
plt.show()

'''calculate correlation and variance'''
print(pearsonr(df_all_years['co2_total'], df_all_years['co2_seasonal']))
print(pearsonr(df_all_years['co2_total'], df_all_years['co2_diurnal']))
print(pearsonr(df_all_years['co2_seasonal'], df_all_years['co2_diurnal']))

print(np.std(df_all_years['co2_total']))
print(np.std(df_all_years['co2_seasonal']))
print(np.std(df_all_years['co2_diurnal']))


