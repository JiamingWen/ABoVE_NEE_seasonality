'''calculate CO2 enhancement based on background values from Carbon Tracker CO2 fields or empirical backgrounds'''

import os
import pandas as pd
import numpy as np

os.chdir('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info

background = 'ct'; variable_name = 'co2_bg_ct'
# background = 'ebg'; variable_name = 'co2_ebg'

for year in [2012, 2013, 2014, 2017]:

    campaign_name = get_campaign_info(year)[2]

    co2_concentration_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_matching_footprint.csv')
    co2_background_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_background-{background}.csv')
    co2_concentration_df['background_CO2'] = co2_background_df[variable_name]
    co2_concentration_df['CO2_change'] = co2_concentration_df['airborne_CO2'] - co2_background_df[variable_name]

    # the above results are for all observations
    # to compare with using background values derived using airborne profiles
    # I filter the data to only include observations below 2000m agl

    co2_concentration_df = co2_concentration_df[co2_concentration_df['footprint_agl'] <= 2000]
    co2_concentration_df = co2_concentration_df.reset_index(drop=True)

    # # make sure the index is the same
    # co2_airborne_profile_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
    # print(np.sum(np.abs(co2_concentration_df['footprint_lat'].values - co2_airborne_profile_df['footprint_lat'].values)))
    # print(np.sum(np.abs(co2_concentration_df['footprint_lon'].values - co2_airborne_profile_df['footprint_lon'].values)))
    # print(np.sum(np.abs(co2_concentration_df['footprint_agl'].values - co2_airborne_profile_df['footprint_agl'].values)))

    co2_concentration_df.to_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change_background-{background}.csv', index=False)