'''compare background values calculated for aircraft data'''
# calculation 1: take average of CO2 concentration measurements above 2,000m agl during each flight date, 
# and use the value as background for all observations during that day

# calculation 2: use the trajectory files to track 500 particles 
# contributing to each observation back where they were located 10 days before the observation, 
# and extract the corresponding CO2 concentration from Carbon Tracker CO2 fields 
# and take the average of 500 particles as background for each observation

# calculation 3: similar to 2, but extracting concentrations from empirical background fields from NOAA GML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info

year = 2012 # 2012 2013 2014 2017
campaign_name = get_campaign_info(year)[2]

data = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
data_background_ct = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change_background-ct.csv')
data_background_ebg = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change_background-ebg.csv')

data['footprint_time_AKT'] = pd.to_datetime(data['footprint_time_AKT'])

# Create a time series plot comparing background CO2 estimates
fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.plot(data['footprint_time_AKT'], data['background_CO2'], label='Airborne Profile', marker='o', linestyle='-', markersize=1, color='k')
ax2.plot(data['footprint_time_AKT'], data_background_ct['background_CO2'], label='Carbon Tracker', marker='s', linestyle='-', markersize=1, color='b')
ax2.plot(data['footprint_time_AKT'], data_background_ebg['background_CO2'], label='Empirical Background', marker='^', linestyle='-', markersize=1, color='r')
ax2.set_xlabel('Time')
fig2.autofmt_xdate()  # Automatically format x-axis labels to avoid overlap
ax2.set_ylabel('Background CO$_2$ (ppm)')
ax2.set_title(f'Background CO$_2$ Time Series: {campaign_name} {year}')
ax2.legend()
plt.tight_layout()
plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_background/compare_background_timeseries_{year}.png', dpi=300)
plt.show()


# Create a scatter plot comparing background CO2 estimates
def create_scatter_plot(ax, x, y, xlabel, ylabel):
    ax.scatter(x, y, s=5, c='blue')
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.axis('equal')
    min_val = 380
    max_val = 415
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1, label='1:1 Line')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Calculate mean bias and RMSE
    mean_bias = np.mean(y - x)
    rmse = np.sqrt(np.mean((y - x)**2))

    # Annotate on plot
    textstr = f"Mean Bias (y-x): {mean_bias:.2f} ppm\nRMSE: {rmse:.2f} ppm"
    ax.text(0.05, 0.2, textstr, transform=ax.transAxes, fontsize=15,
        verticalalignment='top')

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

create_scatter_plot(axs[0], data['background_CO2'], data_background_ct['background_CO2'],
                    'Airborne Profiles (ppm)',
                    'Carbon Tracker (ppm)')

create_scatter_plot(axs[1], data['background_CO2'], data_background_ebg['background_CO2'],
                    'Airborne Profiles (ppm)',
                    'Empirical Background (ppm)')

create_scatter_plot(axs[2], data_background_ct['background_CO2'], data_background_ebg['background_CO2'],
                    'Carbon Tracker (ppm)',
                    'Empirical Background (ppm)')

fig.suptitle(f'{campaign_name} {year}', fontsize=20)
plt.tight_layout()
plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_background/compare_background_scatterplot_{year}.png', dpi=300)
plt.show()
