
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLSResults
import sys
sys.path.append('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info


# whether to filter observations based on land covers they are most sensitive to
lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

df = pd.DataFrame()
for year in [2012, 2013, 2014, 2017]: #2012, 2013, 2014, 2017

    start_month, end_month, campaign_name = get_campaign_info(year)
    month_num = end_month - start_month + 1

    # read atmospheric observations
    df_airborne = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
    df_influence = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')


    # filters for airborne observations
    mask_id = np.where(df_airborne['background_CO2_std'].notna())[0]

    # # land cover filtering 1: select observations with footprint sensitivity of certain land cover > 50%
    # if lcname == 'forest':
    #     mask_id_lc = np.where(df_influence['forest_influence'] / df_influence['total_influence'] > 0.5)[0].tolist()
    #     mask_id = [i for i in mask_id if i in mask_id_lc]
    # elif lcname == 'shrub':
    #     mask_id_lc = np.where(df_influence['shrub_influence'] / df_influence['total_influence'] > 0.5)[0].tolist()
    #     mask_id = [i for i in mask_id if i in mask_id_lc]   
    # elif lcname == 'tundra':
    #     mask_id_lc = np.where(df_influence['tundra_influence'] / df_influence['total_influence'] > 0.5)[0].tolist()
    #     mask_id = [i for i in mask_id if i in mask_id_lc]     

    # land cover filtering 2: select observations with largest footprint sensitivity to certain land cover
    if lcname == 'forest':
        mask_id_lc = np.where((df_influence['forest_influence'] > df_influence['shrub_influence']) & 
                                (df_influence['forest_influence'] > df_influence['tundra_influence']) & 
                                (df_influence['forest_influence'] > df_influence['total_influence'] - df_influence['forest_influence'] - df_influence['shrub_influence'] - df_influence['tundra_influence'])
                                )[0].tolist()
        mask_id = [i for i in mask_id if i in mask_id_lc]
    elif lcname == 'shrub':
        mask_id_lc = np.where((df_influence['shrub_influence'] > df_influence['forest_influence']) & 
                                (df_influence['shrub_influence'] > df_influence['tundra_influence']) & 
                                (df_influence['shrub_influence'] > df_influence['total_influence'] - df_influence['forest_influence'] - df_influence['shrub_influence'] - df_influence['tundra_influence'])
                                )[0].tolist()
        mask_id = [i for i in mask_id if i in mask_id_lc]   
    elif lcname == 'tundra':
        mask_id_lc = np.where((df_influence['tundra_influence'] > df_influence['forest_influence']) & 
                                (df_influence['tundra_influence'] > df_influence['shrub_influence']) & 
                                (df_influence['tundra_influence'] > df_influence['total_influence'] - df_influence['forest_influence'] - df_influence['shrub_influence'] - df_influence['tundra_influence'])
                                )[0].tolist()
        mask_id = [i for i in mask_id if i in mask_id_lc]     

    mask_id_above_influence0 = np.where(df_influence['ABoVE_influence_fraction'] > 0.5)[0]
    mask_id_above_influence = [i for i in mask_id if i in mask_id_above_influence0]

    mask_id_ocean_influence0 = np.where(df_influence['ocean_influence_fraction'] < 0.3)[0]
    mask_id_ocean_influence = [i for i in mask_id if i in mask_id_ocean_influence0]

    mask_id_co2_change0 = np.where(df_airborne['CO2_change'] < 30)[0]
    mask_id_co2_change = [i for i in mask_id if i in mask_id_co2_change0]

    mask_id_co_change0 = np.where(df_airborne['CO_change'] < 40)[0]
    mask_id_co_change = [i for i in mask_id if i in mask_id_co_change0]

    mask_id_final = [i for i in mask_id if i in mask_id_above_influence and i in mask_id_ocean_influence and i in mask_id_co2_change and i in mask_id_co_change]

    # Summarize the length of mask_id vectors for each year
    df = pd.concat([df, pd.DataFrame([{'year': year, 
    'none': len(mask_id),
    'ABoVE': len(mask_id_above_influence),
    'ocean': len(mask_id_ocean_influence),
    'co2': len(mask_id_co2_change),
    'co': len(mask_id_co_change),
    'all': len(mask_id_final)}])], ignore_index=True)


df_sum = df.sum(numeric_only=True).to_frame().T
df_sum['year'] = 'all years'
df = pd.concat([df, df_sum], ignore_index=True)
df.to_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/data_filter/data_filter_summary.csv', index=False)

# barplots
fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharey=True)

categories = ['none', 'ABoVE', 'ocean', 'co2', 'co', 'all']
# labels = ['None', 'ABoVE Influence', 'Ocean Influence', 'CO2 Enhancement', 'CO Enhancement', 'All']
colors = ['lightgrey', 'lightblue', 'lightgreen', 'lightcoral', 'orange', 'skyblue']

for i, year in enumerate([2012, 2013, 2014, 2017]):
    row, col = divmod(i, 2)
    axes[row, col].bar(categories, [df[df['year'] == year][category].values[0] for category in categories], color=colors)
    axes[row, col].set_title(year)
    axes[row, col].set_xlabel('Filter')
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/data_filter/data_filter_summary_individual_years.png')
plt.savefig('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/data_filter/data_filter_summary_individual_years.pdf')
plt.show()


# plot for all years
fig, ax = plt.subplots(figsize=(4, 4))
ax.bar(categories, [df[df['year'] == 'all years'][category].values[0] for category in categories], color=colors)
ax.set_title('All Years')
ax.set_xlabel('Filter')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/data_filter/data_filter_summary_all_years.png')
plt.savefig('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/data_filter/data_filter_summary_all_years.pdf')
plt.show()