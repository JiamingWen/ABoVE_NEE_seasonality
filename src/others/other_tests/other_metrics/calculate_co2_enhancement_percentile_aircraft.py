'''calculate observed and modeled CO2 enhancement percentile for aircraft data during 2012-2014 and 2017'''

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
    df_year_cp = df_year_cp[['obs'] + df_TRENDY.columns.tolist() + df_inversions.columns.tolist() + df_upscaledEC.columns.tolist()]
    df_year_cp_5 = df_year_cp.quantile(0.05)
    df_year_cp_10 = df_year_cp.quantile(0.10)
    df_year_cp_25 = df_year_cp.quantile(0.25)
    df_year_cp_50 = df_year_cp.quantile(0.5)
    df_year_cp_75 = df_year_cp.quantile(0.75)
    df_year_cp_90 = df_year_cp.quantile(0.90)
    df_year_cp_95 = df_year_cp.quantile(0.95)

    df_year_cp_percentiles = pd.concat([
        df_year_cp_5, df_year_cp_10, df_year_cp_25,
        df_year_cp_50, df_year_cp_75, df_year_cp_90,
        df_year_cp_95
    ], axis=1).T

    df_year_cp_percentiles = df_year_cp_percentiles.reset_index().rename(columns={'index': 'percentile'})

    df_year_cp_percentiles.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_percentiles_aircraft_{year}{seasonal}{background}.csv', index=False)

    if year == 2012:
        df_combined = df_year
    else:
        df_combined = pd.concat([df_combined, df_year], ignore_index=True)

df_combined = df_combined[['obs'] + df_TRENDY.columns.tolist() + df_inversions.columns.tolist() + df_upscaledEC.columns.tolist()]
df_combined_5 = df_combined.quantile(0.05)
df_combined_10 = df_combined.quantile(0.10)
df_combined_25 = df_combined.quantile(0.25)
df_combined_50 = df_combined.quantile(0.5)
df_combined_75 = df_combined.quantile(0.75)
df_combined_90 = df_combined.quantile(0.90)
df_combined_95 = df_combined.quantile(0.95)

df_combined_percentiles = pd.concat([
    df_combined_5, df_combined_10, df_combined_25,
    df_combined_50, df_combined_75, df_combined_90,
    df_combined_95
], axis=1).T

df_combined_percentiles = df_combined_percentiles.reset_index().rename(columns={'index': 'percentile'})

df_combined_percentiles.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_percentiles_aircraft{seasonal}{background}.csv', index=False)


'''plot sensonal co2 enhancement amplitude for aircraft data during 2012-2014 and 2017'''
df_combined_percentiles = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_percentiles_aircraft{seasonal}{background}.csv')
df_combined_percentiles_2012 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_percentiles_aircraft_2012{seasonal}{background}.csv')
df_combined_percentiles_2013 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_percentiles_aircraft_2013{seasonal}{background}.csv')
df_combined_percentiles_2014 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_percentiles_aircraft_2014{seasonal}{background}.csv')
df_combined_percentiles_2017 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_percentiles_aircraft_2017{seasonal}{background}.csv')

fig, ax = plt.subplots(figsize=(5, 4))
model_name = 'obs' # obs ISBA-CTRIP CLASSIC IBIS LPJ CLM5.0 CAMS CT-NOAA CMS-Flux ABCflux-NEE
plt.plot(df_combined_percentiles['percentile'], df_combined_percentiles[model_name], label='Multiyear', color='black', marker='o')
plt.plot(df_combined_percentiles_2012['percentile'], df_combined_percentiles_2012[model_name], label='2012', color='blue', marker='o')
plt.plot(df_combined_percentiles_2013['percentile'], df_combined_percentiles_2013[model_name], label='2013', color='green', marker='o')
plt.plot(df_combined_percentiles_2014['percentile'], df_combined_percentiles_2014[model_name], label='2014', color='red', marker='o')
plt.plot(df_combined_percentiles_2017['percentile'], df_combined_percentiles_2017[model_name], label='2017', color='orange', marker='o')
plt.legend()
plt.xlabel('Percentile')
plt.ylabel('CO2 enhancement (ppm)')
plt.xticks([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], labels=['5', '10', '25', '50', '75', '90', '95'])
plt.ylim([-20, 20])
plt.title(f'{model_name}')
plt.show()



'''plot sensonal co2 enhancement amplitude from original data vs mean seasonal cycle'''
df_combined_percentiles = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_percentiles_aircraft.csv')
df_combined_percentiles_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/other_metrics/seasonal_amplitude/seasonal_co2_enhancement_percentiles_aircraft_only_seasonal.csv')

# Define model groups and their colors/shapes
high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM']
upscaled_EC = ['FluxCOM-X-NEE', 'ABCflux-NEE']
inversions = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE']

model_colors = {}
model_shapes = {}

for m in high_skill_TRENDY:
    model_colors[m] = '#396bb8'
    model_shapes[m] = 'o'
for m in low_skill_TRENDY:
    model_colors[m] = '#d4631d'
    model_shapes[m] = 'o'
model_colors['IBIS'] = 'grey'
model_shapes['IBIS'] = 'o'
for m in upscaled_EC:
    model_colors[m] = '#56983f'
    model_shapes[m] = 'd'
for m in inversions:
    model_colors[m] = 'black'
    model_shapes[m] = 's'

# Plot for each percentile
for percentile in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
    plt.figure(figsize=(5, 4))
    for model in df_combined_percentiles.columns:
        if model == 'percentile':
            continue
        if model not in model_colors:
            continue
        x = df_combined_percentiles[df_combined_percentiles['percentile'] == percentile][model].values
        y = df_combined_percentiles_only_seasonal[df_combined_percentiles_only_seasonal['percentile'] == percentile][model].values
        plt.scatter(x, y, marker=model_shapes[model], color=model_colors[model], facecolor='none', s=30, label=model)
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([-20, 20], [-20, 20], color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Original data')
    plt.ylabel('Mean seasonal cycle')
    plt.title(f'CO2 enhancement percentile {percentile}')
    plt.show()