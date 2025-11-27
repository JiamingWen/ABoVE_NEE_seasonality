'''compare CO2 enhancement due to ocean fluxes from different datasets'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
import sys
sys.path.append('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info

for year in [2012, 2013, 2014, 2017]: #2012, 2013, 2014, 2017

    start_month, end_month, campaign_name = get_campaign_info(year)
    month_num = end_month - start_month + 1

    df_ocean_year = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_ocean.csv')
    df_obs_year = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
    df_fire_year = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fire.csv')
    df_fossil_year = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fossil.csv')
    df_land_year = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_inversions.csv')

    if year == 2012:
        df_ocean = df_ocean_year
        df_obs = df_obs_year
        df_fire = df_fire_year
        df_fossil = df_fossil_year
        df_land = df_land_year
    else:
        df_ocean = pd.concat((df_ocean, df_ocean_year), ignore_index=True)
        df_obs = pd.concat((df_obs, df_obs_year), ignore_index=True)
        df_fire = pd.concat((df_fire, df_fire_year), ignore_index=True)
        df_fossil = pd.concat((df_fossil, df_fossil_year), ignore_index=True)
        df_land = pd.concat((df_land, df_land_year), ignore_index=True)


'''Plot time series for each column in df_ocean'''
plt.figure(figsize=(8, 5))
for column in df_ocean.columns:
    plt.scatter(df_ocean.index, df_ocean[column], label=column)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Index', fontsize=14)
plt.ylabel('CO$_2$ enhancement (ppm)', fontsize=14)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.grid()
plt.tight_layout()
plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_ocean_fluxes/compare_co2_enhancement_ocean_fluxes.png')
plt.show()


'''Calculate and plot cross-correlation among data products'''
columns = df_ocean.columns

# Create a heatmap for the cross-correlation matrix
correlation_matrix = pd.DataFrame(index=columns, columns=columns)

for col1 in columns:
    for col2 in columns:
        if col1 == col2:
            correlation_matrix.loc[col1, col2] = 1.0
        else:
            correlation_matrix.loc[col1, col2] = pearsonr(df_ocean[col1], df_ocean[col2])[0]

correlation_matrix = correlation_matrix.astype(float)

plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm_r', interpolation='nearest', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks(ticks=np.arange(len(columns)), labels=columns, rotation=45, fontsize=10)
plt.yticks(ticks=np.arange(len(columns)), labels=columns, fontsize=10)
for i in range(len(columns)):
    for j in range(len(columns)):
        plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=12)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_ocean_fluxes/compare_co2_enhancement_ocean_fluxes_cor.png')
plt.show()


'''Plot time series for different fluxes in separate panels'''
fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

# Observed
axs[0].scatter(df_obs.index, df_obs['CO2_change'], color='black')
axs[0].text(0.02, 0.9, 'Observed', transform=axs[0].transAxes, fontsize=18, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
# axs[0].set_ylabel('CO$_2$ enhancement (ppm)', fontsize=12)
axs[0].grid()
axs[0].tick_params(axis='y', labelsize=12)
axs[0].set_ylim(-40, 60)

# Land
axs[1].scatter(df_land.index, df_land['CT-NOAA'], color='green')
axs[1].text(0.02, 0.9, 'Land', transform=axs[1].transAxes, fontsize=18, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
# axs[1].set_ylabel('CO$_2$ enhancement (ppm)', fontsize=12)
axs[1].grid()
axs[1].tick_params(axis='y', labelsize=12)
axs[1].set_ylim(-40, 60)

# Ocean
axs[2].scatter(df_ocean.index, df_ocean.mean(axis=1), color='blue')
axs[2].text(0.02, 0.9, 'Ocean', transform=axs[2].transAxes, fontsize=18, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
axs[2].set_ylabel('CO$_2$ enhancement (ppm)', fontsize=15)
axs[2].grid()
axs[2].tick_params(axis='y', labelsize=12)
axs[2].set_ylim(-40, 60)

# Fossil
axs[3].scatter(df_fossil.index, df_fossil['odiac2022'], color='orange')
axs[3].text(0.02, 0.9, 'Fossil', transform=axs[3].transAxes, fontsize=18, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
# axs[3].set_ylabel('CO$_2$ enhancement (ppm)', fontsize=12)
axs[3].grid()
axs[3].tick_params(axis='y', labelsize=12)
axs[3].set_ylim(-40, 60)

# Fire
axs[4].scatter(df_fire.index, df_fire['gfed4.1'], color='red')
axs[4].text(0.02, 0.9, 'Fire', transform=axs[4].transAxes, fontsize=18, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
# axs[4].set_ylabel('CO$_2$ enhancement (ppm)', fontsize=12)
axs[4].set_xlabel('Index', fontsize=14)
axs[4].grid()
axs[4].tick_params(axis='y', labelsize=12)
axs[4].tick_params(axis='x', labelsize=12)
axs[4].set_ylim(-40, 60)

plt.tight_layout()
plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_ocean_fluxes/compare_co2_enhancement_all_fluxes.png')
plt.show()