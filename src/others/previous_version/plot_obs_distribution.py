# plot statistical distribution of observations, e.g., month, time of day, land covers, etc.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info


month_list = np.arange(4,12)
obs_dist_month_df = pd.DataFrame(columns=['year'] + [str(i) for i in month_list])

hour_list = np.arange(9,19)
obs_dist_hour_df = pd.DataFrame(columns=['year'] + [str(i) for i in hour_list])

obs_dist_lc_df = pd.DataFrame(columns=['year', 'forest', 'shrub', 'tundra', 'others'])

num = -1
for year in [2012, 2013, 2014, 2017]:
    num += 1
    start_month, end_month, campaign_name = get_campaign_info(year)

    # read observations
    df_airborne = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
    df_influence = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')
    df = pd.concat((df_airborne, df_influence), axis=1)
    n_receptor = df.shape[0]


    # filters for airborne observations
    mask_id = np.where((df['background_CO2_std'].notna()) &
        # (local_hour.isin([13, 14, 15, 16])) &
        (df['ABoVE_influence_fraction'] > 0.5) &
        (df['ocean_influence_fraction'] < 0.3) &
        # (df['ABoVE_land_influence_fraction'] > 0.5)) and
        (df['CO2_change'] < 30) &
        (df['CO_change'] < 40))[0]

    df_subset = df.loc[mask_id]

    # df_subset = df

    local_month = pd.to_datetime(df_subset['footprint_time_AKT'], utc=True).dt.tz_convert('America/Anchorage').dt.month
    local_hour = pd.to_datetime(df_subset['footprint_time_AKT'], utc=True).dt.tz_convert('America/Anchorage').dt.hour
    
    month_count = [year]
    for month in month_list:
        month_count.append(np.sum(local_month == month))
    obs_dist_month_df.loc[num] = month_count

    hour_count = [year]
    for hour in hour_list:
        hour_count.append(np.sum(local_hour == hour))
    obs_dist_hour_df.loc[num] = hour_count

    lc_count = [year]
    df_subset['other_influence'] = df_subset.total_influence -df_subset.forest_influence -df_subset.shrub_influence -df_subset.tundra_influence
    df_tmp = df_subset[['forest_influence', 'shrub_influence', 'tundra_influence', 'other_influence']]
    lc_dominant = df_tmp.apply(np.argmax, axis=1)
    for lc in [0, 1, 2, 3]: # forst, shrub, tundra, others
        lc_count.append(np.sum(lc_dominant == lc))
    obs_dist_lc_df.loc[num] = lc_count

    # plt.bar(local_month)
    # plt.xlim(4,11)
    # plt.show()

    # plot CO vs CO2 enhancement
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_subset.CO_change, df_subset.CO2_change, c=local_month, cmap="Spectral")
    plt.xlabel('CO enhancement (ppb)')
    plt.ylabel('CO2 enhancement (ppm)')
    legend1 = ax.legend(*scatter.legend_elements(),title='Month')
    ax.add_artist(legend1)
    plt.title(year)
    plt.show()

obs_dist_month_df.plot(x='year', kind='bar', stacked=True,
    title='Distribution of Months')
plt.show()

obs_dist_hour_df.plot(x='year', kind='bar', stacked=True,
    title='Distribution of Time of Day')
plt.show()

obs_dist_lc_df.plot(x='year', kind='bar', stacked=True,
    title='Distribution of LC that obs is most sensitive to')
plt.show()