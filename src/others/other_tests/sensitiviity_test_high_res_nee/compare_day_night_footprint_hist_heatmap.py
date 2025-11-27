'''
Compare if daytime observations are more sensitive to daytime flux or not
To do so, I compare daytime and nighttime footprints for each observation
Ideally, I can use the hourly footprint, but it took longer time to run;
here I utilize the reprocessed 3-hourly footprint, aggregated during daytime and nighttime 9pm-9am
Alaska Standard Time UTC-9
daytime: local time  9am-9pm, UTC time 6pm-6am
nighttime: local time 9pm-9am, UTC time 6am-6pm
'''

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import sys
sys.path.append('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
import utils
from functions import get_campaign_info


# '''extract 3-hour sum of footprints summed over all land pixels, for each observation'''
# year = 2014 # 2012 2013 2014 2017

# campaign_name = get_campaign_info(year)[2]
# config = utils.getConfig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/config/config_{campaign_name}{year}_3hourly.ini')

# # read observations
# receptor_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
# n_receptor = receptor_df.shape[0]


# # mask for land pixels
# cell_id_table = pd.read_csv('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/cell_id_table/cell_id_table.csv')
# land_cellnum_list = np.where(cell_id_table['land']==1)[0]

# # create a data frame for output
# output_df = pd.DataFrame(
#     0.0,
#     index=np.arange(n_receptor),
#     columns=[f"UTC_hour_{hour}" for hour in range(0, 24, 3)]
# )

# # read H matrix
# config = utils.getConfig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/config/config_{campaign_name}{year}_3hourly.ini')
# for ntimestep in np.arange(0, config["ntimesteps"]):
    
#     print(ntimestep)
#     timestep = config["sdate"] + ntimestep * config["timestep"]
#     h_matrix_dir= f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/3hourly'
#     h_matrix_file = f'{h_matrix_dir}/H{timestep.year}_{timestep.month}_{timestep.day}_{timestep.hour}.txt'
    
#     if os.path.exists(h_matrix_file):
#         # print(f"Reading {h_matrix_file}")
#         h_df = pd.read_csv(
#             h_matrix_file,
#             sep="\s+", index_col=False, header=None,
#             names=["obs_id", "cell_id", "lat_id", "lon_id", "lat", "lon", "val"]
#         )
#         #  \s+ is the expression for "any amount of whitespace"

#         # Create sparse matrix directly
#         n_cell = 720 * 120
#         h_matrix0 = csr_matrix((h_df.val, (h_df.obs_id, h_df.cell_id)), shape=(n_receptor, n_cell))

#     else: # no footprint falls in this time period
#         h_matrix0 = csr_matrix((n_receptor, 720 * 120))

#     # only account for land pixels
#     h_matrix0_subset = h_matrix0[:, land_cellnum_list]
#     del h_matrix0

#     influence = h_matrix0_subset.sum(axis=1)
#     del h_matrix0_subset

#     output_df[f'UTC_hour_{timestep.hour}'] += np.array(influence).flatten()

# dir_output = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/diurnal_distribution'
# os.makedirs(dir_output, exist_ok=True)
# output_df.to_csv(f'{dir_output}/footprint_diurnal_distribution_{campaign_name}_{year}_3hourly.csv', index=False)


'''plot daytime - nighttime footprint difference'''
for year in [2012, 2013, 2014, 2017]:
    campaign_name = get_campaign_info(year)[2]
    dir_input = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/diurnal_distribution'
    df_footprint = pd.read_csv(f'{dir_input}/footprint_diurnal_distribution_{campaign_name}_{year}_3hourly.csv')

    # read atmospheric observations
    df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
    df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

    # filters for airborne observations
    mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
        (df_influence['ABoVE_influence_fraction'] > 0.5) &
        (df_influence['ocean_influence_fraction'] < 0.3) &
        (df_airborne['CO2_change'] < 30) &
        (df_airborne['CO_change'] < 40))[0]

    df_footprint = pd.concat([df_footprint, df_airborne], axis=1)
    df_footprint = df_footprint.iloc[mask_id]

    # daytime: local time 9am-9pm, UTC time 6pm-6am
    # night time: local time 9pm-9am, UTC time 6am-6pm
    daytime_hours = [18, 21, 0, 3]
    nighttime_hours = [6, 9, 12, 15]

    # # daytime: local time 6am-6pm, UTC time 3pm-3am
    # # night time: local time 6pm-6am, UTC time 3am-3pm
    # daytime_hours = [15, 18, 21, 0]
    # nighttime_hours = [3, 6, 9, 12]

    df_footprint['daytime_footprint'] = df_footprint[[f'UTC_hour_{hour}' for hour in daytime_hours]].sum(axis=1)
    df_footprint['nighttime_footprint'] = df_footprint[[f'UTC_hour_{hour}' for hour in nighttime_hours]].sum(axis=1)
    df_footprint['alltime_footprint'] = df_footprint[[f'UTC_hour_{hour}' for hour in daytime_hours + nighttime_hours]].sum(axis=1)
    df_footprint['day_night_diff'] = df_footprint['daytime_footprint'] - df_footprint['nighttime_footprint']
    df_footprint['day_night_diff_frac'] = df_footprint['day_night_diff'] / df_footprint['alltime_footprint']

    # plot histogram for difference
    plt.figure(figsize=(6,4))
    plt.hist(df_footprint['day_night_diff'], bins=np.arange(-15, 16, 1), color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title(f'{campaign_name} {year}', fontsize=16)
    plt.xlabel('Daytime - Nighttime Footprint Difference\nppm $(\\mu mol\\ m^{-2}\\ s^{-1})^{-1}$', fontsize=14)
    plt.ylabel('Number of Observations', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-15, 15)

    mean_diff = df_footprint['day_night_diff'].mean()
    plt.text(0.05, 0.95, f"mean: {mean_diff:.2f}", transform=plt.gca().transAxes,
             fontsize=16, verticalalignment='top', color='black')

    plt.savefig(f'{dir_input}/day_night_footprint_difference_histogram_{campaign_name}_{year}.png', dpi=300)
    plt.show()


    # plot histogram for difference percentage
    plt.figure(figsize=(6,4))
    plt.hist(df_footprint['day_night_diff_frac']*100, bins=np.arange(-100, 100, 5), color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title(f'{campaign_name} {year}', fontsize=16)
    plt.xlabel('Daytime - Nighttime Footprint Difference (%)', fontsize=14)
    plt.ylabel('Number of Observations', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-100, 100)

    mean_diff_frac = df_footprint['day_night_diff_frac'].mean()
    plt.text(0.05, 0.95, f"mean: {int(mean_diff_frac*100)}%", transform=plt.gca().transAxes,
             fontsize=16, verticalalignment='top', color='black')

    plt.savefig(f'{dir_input}/day_night_footprint_difference_percent_histogram_{campaign_name}_{year}.png', dpi=300)
    plt.show()


    if year == 2012:
        df_footprint_all = df_footprint
    else:
        df_footprint_all = pd.concat([df_footprint_all, df_footprint], axis=0)


'''analysis for difference'''
# histogram
plt.figure(figsize=(6,4))
plt.hist(df_footprint_all['day_night_diff'], bins=np.arange(-15, 16, 1), color='skyblue', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('Daytime - Nighttime Footprint Difference\nppm $(\\mu mol\\ m^{-2}\\ s^{-1})^{-1}$', fontsize=14)
plt.ylabel('Number of Observations', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-15, 15)
mean_diff = df_footprint_all['day_night_diff'].mean()
plt.text(0.05, 0.95, f"mean: {mean_diff:.2f}", transform=plt.gca().transAxes,
            fontsize=16, verticalalignment='top', color='black')
plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/day_night_footprint_difference_histogram.png', dpi=300)
plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/day_night_footprint_difference_histogram.pdf', dpi=300)
plt.show()


#plot the difference in the function of other variables
df_footprint_all['local_hour'] = df_footprint_all['footprint_time_AKT'].astype(str).str[11:13].astype(int)
df_footprint_all['local_month'] = df_footprint_all['footprint_time_AKT'].astype(str).str[5:7].astype(int)

#plot difference vs. local hour x month
heatmap_data = df_footprint_all.pivot_table(index='local_month', columns='local_hour', values='day_night_diff', aggfunc='mean')
count_data = df_footprint_all.pivot_table(index='local_month', columns='local_hour', values='day_night_diff', aggfunc='count')
heatmap_data[count_data < 10] = np.nan
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm', origin='lower', vmin=-3, vmax=3)
plt.colorbar(label='Daytime - Nighttime Footprint Difference\nppm $(\\mu mol\\ m^{-2}\\ s^{-1})^{-1}$')
plt.xticks(ticks=np.arange(heatmap_data.columns.size), labels=heatmap_data.columns, fontsize=12)
plt.yticks(ticks=np.arange(heatmap_data.index.size), labels=heatmap_data.index, fontsize=12)
plt.xlabel('Local Hour', fontsize=13)
plt.ylabel('Month', fontsize=13)
plt.show()


#plot difference vs. agl x month
df_footprint_all['footprint_agl_bin'] = pd.cut(df_footprint_all['footprint_agl'], bins=np.arange(0, df_footprint_all['footprint_agl'].max() + 100, 100))
heatmap_data = df_footprint_all.pivot_table(index='footprint_agl_bin', columns='local_month', values='day_night_diff', aggfunc='mean')
count_data = df_footprint_all.pivot_table(index='footprint_agl_bin', columns='local_month', values='day_night_diff', aggfunc='count')
heatmap_data[count_data < 10] = np.nan
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm', origin='lower', vmin=-3, vmax=3)
plt.colorbar(label='Daytime - Nighttime Footprint Difference\nppm $(\\mu mol\\ m^{-2}\\ s^{-1})^{-1}$')
plt.xticks(ticks=np.arange(heatmap_data.columns.size), labels=heatmap_data.columns, fontsize=12)
plt.yticks(ticks=np.arange(heatmap_data.index.size), labels=heatmap_data.index, fontsize=12)
plt.xlabel('Month', fontsize=13)
plt.ylabel('Above Ground Level (m)', fontsize=13)
plt.show()


#plot difference vs. local hour x agl
heatmap_data = df_footprint_all.pivot_table(index='footprint_agl_bin', columns='local_hour', values='day_night_diff', aggfunc='mean')
count_data = df_footprint_all.pivot_table(index='footprint_agl_bin', columns='local_hour', values='day_night_diff', aggfunc='count')
heatmap_data[count_data < 10] = np.nan
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm', origin='lower', vmin=-3, vmax=3)
plt.colorbar(label='Daytime - Nighttime Footprint Difference\nppm $(\\mu mol\\ m^{-2}\\ s^{-1})^{-1}$')
plt.xticks(ticks=np.arange(heatmap_data.columns.size), labels=heatmap_data.columns, fontsize=12)
plt.yticks(ticks=np.arange(heatmap_data.index.size), labels=heatmap_data.index, fontsize=12)
plt.xlabel('Local Hour', fontsize=13)
plt.ylabel('Above Ground Level (m)', fontsize=13)
plt.show()


'''analysis for difference percentage'''
# plot histogram for difference percentage
plt.figure(figsize=(6,4))
plt.hist(df_footprint_all['day_night_diff_frac']*100, bins=np.arange(-100, 100, 5), color='skyblue', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('Daytime - Nighttime Footprint Difference (%)', fontsize=14)
plt.ylabel('Number of Observations', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-100, 100)
mean_diff_frac = df_footprint_all['day_night_diff_frac'].mean()
plt.text(0.05, 0.95, f"mean: {int(mean_diff_frac*100)}%", transform=plt.gca().transAxes,
            fontsize=16, verticalalignment='top', color='black')
plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/day_night_footprint_difference_percent_histogram.png', dpi=300)
plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/day_night_footprint_difference_percent_histogram.pdf', dpi=300)
plt.show()


#plot the difference percentage in the function of other variables
df_footprint_all['local_hour'] = df_footprint_all['footprint_time_AKT'].astype(str).str[11:13].astype(int)
df_footprint_all['local_month'] = df_footprint_all['footprint_time_AKT'].astype(str).str[5:7].astype(int)

#plot difference vs. local hour x month'''
heatmap_data = df_footprint_all.pivot_table(index='local_month', columns='local_hour', values='day_night_diff_frac', aggfunc='mean')
count_data = df_footprint_all.pivot_table(index='local_month', columns='local_hour', values='day_night_diff_frac', aggfunc='count')
heatmap_data[count_data < 10] = np.nan
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data*100, aspect='auto', cmap='coolwarm', origin='lower', vmin=-25, vmax=25)
plt.colorbar(label='Daytime - Nighttime Footprint Difference (%)')
plt.xticks(ticks=np.arange(heatmap_data.columns.size), labels=heatmap_data.columns, fontsize=12)
plt.yticks(ticks=np.arange(heatmap_data.index.size), labels=heatmap_data.index, fontsize=12)
plt.xlabel('Local Hour', fontsize=13)
plt.ylabel('Month', fontsize=13)
plt.show()


#plot difference vs. agl x month
df_footprint_all['footprint_agl_bin'] = pd.cut(df_footprint_all['footprint_agl'], bins=np.arange(0, df_footprint_all['footprint_agl'].max() + 100, 100))
heatmap_data = df_footprint_all.pivot_table(index='footprint_agl_bin', columns='local_month', values='day_night_diff_frac', aggfunc='mean')
count_data = df_footprint_all.pivot_table(index='footprint_agl_bin', columns='local_month', values='day_night_diff_frac', aggfunc='count')
heatmap_data[count_data < 10] = np.nan
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data*100, aspect='auto', cmap='coolwarm', origin='lower', vmin=-25, vmax=25)
plt.colorbar(label='Daytime - Nighttime Footprint Difference (%)')
plt.xticks(ticks=np.arange(heatmap_data.columns.size), labels=heatmap_data.columns, fontsize=12)
plt.yticks(ticks=np.arange(heatmap_data.index.size), labels=heatmap_data.index, fontsize=12)
plt.xlabel('Month', fontsize=13)
plt.ylabel('Above Ground Level (m)', fontsize=13)
plt.show()


#plot difference vs. local hour x agl
heatmap_data = df_footprint_all.pivot_table(index='footprint_agl_bin', columns='local_hour', values='day_night_diff_frac', aggfunc='mean')
count_data = df_footprint_all.pivot_table(index='footprint_agl_bin', columns='local_hour', values='day_night_diff_frac', aggfunc='count')
heatmap_data[count_data < 10] = np.nan
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data*100, aspect='auto', cmap='coolwarm', origin='lower', vmin=-25, vmax=25)
plt.colorbar(label='Daytime - Nighttime Footprint Difference (%)')
plt.xticks(ticks=np.arange(heatmap_data.columns.size), labels=heatmap_data.columns, fontsize=12)
plt.yticks(ticks=np.arange(heatmap_data.index.size), labels=heatmap_data.index, fontsize=12)
plt.xlabel('Local Hour', fontsize=13)
plt.ylabel('Above Ground Level (m)', fontsize=13)
plt.show()


# Plot the counts of each AGL bin using a bar plot
bin_counts = df_footprint_all['footprint_agl_bin'].value_counts().sort_index()
plt.figure(figsize=(7, 4))
plt.bar(bin_counts.index.astype(str), bin_counts.values)
plt.xlabel('Above Ground Level (m) Bin')
plt.ylabel('Number of Observations')
plt.title('Distribution of Observations by AGL Bin')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Plot the counts of each local hour using a bar plot
hour_counts = df_footprint_all['local_hour'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(hour_counts.index.astype(str), hour_counts.values)
plt.xlabel('Local Hour')
plt.ylabel('Number of Observations')
plt.title('Distribution of Observations by Local Hour')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()