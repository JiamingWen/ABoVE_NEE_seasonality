'''compare the background calculations using 2,000 m or 3,000 m as cutoff values'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info
from matplotlib.ticker import MaxNLocator

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for year_id, year in enumerate([2012, 2013, 2014, 2017]):
    campaign_name = get_campaign_info(year)[2]

    # data = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_matching_footprint.csv')
    # data_3000m = data[data['footprint_agl']>3000]

    # # extract unique dates with measurements >3000 agl 
    # datelist = []
    # unique_datelist = []
    # for index, row in data_3000m.iterrows():
    #     date = pd.Timestamp(row['footprint_time_AKT']).date()
    #     datelist.append(date)
    #     if date not in unique_datelist:
    #         unique_datelist.append(date)

    # # for each date, take the average of all the CO2 concentration measurements during that day
    # background_df = pd.DataFrame([])
    # for date in unique_datelist:
    #     print(date)
    #     indices = [i for i, x in enumerate(datelist) if x == date]
    #     tmp_CO2 = [data_3000m.iloc[[i]]['airborne_CO2'] for i in indices]
    #     CO2 = np.nanmean(tmp_CO2)
    #     CO2_std = np.nanstd(tmp_CO2)
    #     tmp_CO = [data_3000m.iloc[[i]]['airborne_CO'] for i in indices]
    #     CO = np.nanmean(tmp_CO)
    #     CO_std = np.nanstd(tmp_CO)
    #     n = len(tmp_CO2)

    #     single_date = pd.DataFrame({'date': date,
    #                                 'CO2': CO2,
    #                                 'CO2_std': CO2_std,
    #                                 'CO': CO,
    #                                 'CO_std': CO_std,
    #                                 'n': n}, index=[0])
    #     background_df = pd.concat([background_df, single_date])

    # background_df.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_background_3000m.csv', encoding='utf-8', index=False)
    
    # read data
    background_df_3000m = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_background_3000m.csv')
    background_df_2000m = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_background.csv')

    # Merge 2000 m and 3000 m background data on date
    background_df_3000m['date'] = pd.to_datetime(background_df_3000m['date'])
    background_df_2000m['date'] = pd.to_datetime(background_df_2000m['date'])

    merged_df = pd.merge(
        background_df_2000m,
        background_df_3000m,
        on='date',
        how='outer',
        suffixes=('_2000m', '_3000m')
    ).sort_values('date')

    ax = axes[year_id]
    ax.plot(merged_df['date'], merged_df['CO2_2000m'], c='black', label='2,000 m')
    ax.plot(merged_df['date'], merged_df['CO2_3000m'], c='red', label='3,000 m')
    ax.set_xlim([datetime(year, 4, 1), datetime(year, 11, 30)])
    ax.set_xticks([datetime(year, m, 1) for m in range(4, 12)])
    ax.set_xticklabels([str(m) for m in range(4, 12)], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    panel_labels = ['a', 'b', 'c', 'd']
    label = panel_labels[year_id]
    ax.text(0.95, 0.95, f'({label}) {year}', transform=ax.transAxes,
            ha='right', va='top', fontsize=16)
    if year_id == 0:
        ax.legend(loc='best', bbox_to_anchor=(0.4, 0.8), fontsize=14)

fig.supxlabel('Month', fontsize=18)
fig.supylabel('CO$_2$ background (ppm)', fontsize=18)
plt.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_background/compare_background_2k_3k.png', dpi=300, bbox_inches='tight')
plt.show()