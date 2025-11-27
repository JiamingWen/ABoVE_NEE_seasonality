'''calculate observed and modeled CO2 enhancement amplitude for aircraft data during 2012-2014 and 2017'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

background = '' # '' '_background-ct' '_background-ebg'
seasonal = '' # '' _only_seasonal

for year in [2012, 2013, 2014, 2017]:

    if year in [2012, 2013, 2014]:
        campaign_name = 'carve'
    elif year == 2017:
        campaign_name = 'arctic_cap'

    # read atmospheric observations
    df_airborne_original = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
    df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change{background}.csv')
    df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

    # influence from fossil and fire emissions
    df_fossil = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fossil.csv')
    df_fire = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fire.csv')

    # observed CO2 enhancement
    df_airborne['obs'] = df_airborne['CO2_change'] - df_fossil['odiac2022'] - df_fire['gfed4.1']

    # modeled CO2 enhancement
    df_TRENDY = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_TRENDYv11{seasonal}.csv')
    if seasonal == '_only_seasonal':
        df_inversions = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_inversionsNEE{seasonal}.csv')
    else:
        df_inversions = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_inversions{seasonal}.csv')
        df_inversions = df_inversions.subtract(df_fire['gfed4.1'], axis=0)  # account for fire emissions
    df_upscaledEC = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_UpscaledEC{seasonal}.csv')

    df_year = pd.concat([df_airborne, df_TRENDY, df_inversions, df_upscaledEC], axis=1)

    # filters for airborne observations
    mask_id = np.where((df_airborne_original['background_CO2_std'].notna()) &
        (df_influence['ABoVE_influence_fraction'] > 0.5) &
        (df_influence['ocean_influence_fraction'] < 0.3) &
        (df_airborne_original['CO2_change'] < 30) &
        (df_airborne_original['CO_change'] < 40))[0]

    df_year = df_year.loc[mask_id]

    df_year_cp = df_year.copy()
    df_year_cp['footprint_time_UTC'] = pd.to_datetime(df_year_cp['footprint_time_UTC'])
    df_year_cp['month'] = df_year_cp['footprint_time_UTC'].dt.month
    df_year_cp = df_year_cp[['month', 'obs'] + df_TRENDY.columns.tolist() + df_inversions.columns.tolist() + df_upscaledEC.columns.tolist()]
    df_year_cp_monthly = df_year_cp.groupby('month').mean().reset_index()
    df_year_cp_monthly.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_amplitude_aircraft_{year}{seasonal}{background}.csv', index=False)

    if year == 2012:
        df_combined = df_year
    else:
        df_combined = pd.concat([df_combined, df_year], ignore_index=True)


df_combined['footprint_time_UTC'] = pd.to_datetime(df_combined['footprint_time_UTC'])
df_combined['month'] = df_combined['footprint_time_UTC'].dt.month
df_combined = df_combined[['month', 'obs'] + df_TRENDY.columns.tolist() + df_inversions.columns.tolist() + df_upscaledEC.columns.tolist()]
df_combined_monthly = df_combined.groupby('month').mean().reset_index()

df_combined_monthly.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_amplitude_aircraft{seasonal}{background}.csv', index=False)


'''plot sensonal co2 enhancement amplitude for aircraft data during 2012-2014 and 2017'''
df_combined_monthly = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_amplitude_aircraft{seasonal}{background}.csv')
df_combined_monthly_2012 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_amplitude_aircraft_2012{seasonal}{background}.csv')
df_combined_monthly_2013 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_amplitude_aircraft_2013{seasonal}{background}.csv')
df_combined_monthly_2014 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_amplitude_aircraft_2014{seasonal}{background}.csv')
df_combined_monthly_2017 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_amplitude_aircraft_2017{seasonal}{background}.csv')

fig, ax = plt.subplots(figsize=(5, 4))
model_name = 'obs' # obs ISBA-CTRIP CLASSIC IBIS LPJ CLM5.0 CAMS CT-NOAA CMS-Flux X-BASE
plt.plot(df_combined_monthly['month'], df_combined_monthly[model_name], label='Multiyear', color='black', marker='o')
plt.plot(df_combined_monthly_2012['month'], df_combined_monthly_2012[model_name], label='2012', color='blue', marker='o')
plt.plot(df_combined_monthly_2013['month'], df_combined_monthly_2013[model_name], label='2013', color='green', marker='o')
plt.plot(df_combined_monthly_2014['month'], df_combined_monthly_2014[model_name], label='2014', color='red', marker='o')
plt.plot(df_combined_monthly_2017['month'], df_combined_monthly_2017[model_name], label='2017', color='orange', marker='o')
plt.legend()
plt.xlabel('Month')
plt.ylabel('CO2 enhancement (ppm)')
plt.ylim([-20, 20])
plt.title(f'{model_name}')
plt.show()
