'''plot footprint sensitivity for each year and each month
summarize the fraction of influence from different land cover types'''

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd

import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info

obs_mode = 'all' # all selected
calculation_method = 'mean'

if obs_mode == 'all':
    filestr = ''
elif obs_mode == 'selected':
    filestr = '_selected'

def plot_influence_map(ds_influence, title, output_path):

    fig = plt.figure(figsize=(9,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # ax[subplot_id].set_extent([-170, -100, 50, 75], ccrs.PlateCarree())
    ax.set_extent([-170, -80, 50, 80], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True) #linewidth=2, color='gray', alpha=0.5, linestyle='--'
    ax.coastlines()
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 15}
    gl.ylabel_style = {'fontsize': 15}
    ax.set_aspect(2)
    plt.title(title, fontsize=20)

    lons = ds_influence["longitude"].values
    lats = ds_influence["latitude"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cax = fig.add_axes([0.15, 0.01, 0.7, 0.04])
    cp = ax.pcolormesh(lon_grid, lat_grid, np.log10(ds_influence), vmin=-4, vmax=-2, cmap='Purples')
    cb = fig.colorbar(cp, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=15)
    cb.set_label(r"ppm $(\mu mol\ m^{-2}\ s^{-1})^{-1}$", fontsize=15, labelpad=10)
    cb.set_ticks([-4, -3, -2])
    cb.set_ticklabels([ r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'])

    plt.show()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')


for year in [2012,2013,2014,2017]:

    start_month, end_month, campaign_name = get_campaign_info(year)

    '''plot footprint sensitivity for each month'''

    for month in np.arange(start_month, end_month+1):
        print(month)
        ds_influence_month = xr.open_dataset(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity/influence_{calculation_method}{year}_{month}{filestr}.nc').influence
        plot_influence_map(ds_influence_month, f'{year}/{month:02d} {obs_mode} observations', f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity/influence_{calculation_method}{year}_{month}{filestr}.png')

    ds_influence_year = xr.open_dataset(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity/influence_{calculation_method}{year}{filestr}.nc').influence
    plot_influence_map(ds_influence_year, f'{year} {obs_mode} observations', f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity/influence_{calculation_method}{year}{filestr}.png')


    # calculate fraction of each land cover and ABoVE core vs extended region
    ds_influence_year = ds_influence_year.isel(
        latitude=(ds_influence_year.latitude >= 30) & (ds_influence_year.latitude <= 90),
        )
    influence_vec = ds_influence_year.values.flatten()
    cell_id_table = pd.read_csv('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/cell_id_table/cell_id_table.csv')

    allregion_cellnum_list = cell_id_table['cell_id']
    ABoVEcore_cellnum_list = np.where(cell_id_table['ABoVE'] == 0)[0]
    ABoVEextended_cellnum_list = np.where(cell_id_table['ABoVE'] == 1)[0]
    ABoVEout_cellnum_list = np.where(cell_id_table['ABoVE'] == 255)[0]

    land_cellnum_list = np.where(cell_id_table['land'] == 1)[0]
    ocean_cellnum_list = np.where(cell_id_table['land'] == 0)[0]
    nonABoVEland_list = np.intersect1d(ABoVEout_cellnum_list, land_cellnum_list)

    alllc_cellnum_list = cell_id_table['cell_id']
    forest_cellnum_list = np.where(cell_id_table['lc'] == 5)[0]
    shrub_cellnum_list = np.where(cell_id_table['lc'] == 7)[0]
    tundra_cellnum_list = np.where(cell_id_table['lc'].isin([8,9,10]))[0]
    otherlc_cellnum_list = np.where(~cell_id_table['lc'].isin([5,7,8,9,10]))[0]

    total_influence = np.sum(influence_vec)

    ABoVEcore_influence = np.sum(influence_vec[ABoVEcore_cellnum_list])
    ABoVEextended_influence = np.sum(influence_vec[ABoVEextended_cellnum_list])
    ocean_influence = np.sum(influence_vec[ocean_cellnum_list])
    nonABoVEland_influence = np.sum(influence_vec[nonABoVEland_list])

    region_names = ['allregion','ABoVEcore', 'ABoVEextended', 'nonABoVEland', 'ocean']
    region_cellnum_lists = [allregion_cellnum_list, ABoVEcore_cellnum_list, ABoVEextended_cellnum_list, nonABoVEland_list, ocean_cellnum_list]
    lc_names = ['alllc', 'forest', 'shrub', 'tundra', 'otherlc']
    lc_cellnum_lists = [alllc_cellnum_list, forest_cellnum_list, shrub_cellnum_list, tundra_cellnum_list, otherlc_cellnum_list]

    influence_summary_df = pd.DataFrame([])
    for region_name, region_cellnum_list in zip(region_names, region_cellnum_lists):

        region_influence_summary_df = pd.DataFrame([])
        for lc_name, lc_cellnum_list in zip(lc_names, lc_cellnum_lists):
            subregion_cellnum_list = np.intersect1d(region_cellnum_list, lc_cellnum_list)
            subregion_influence = np.sum(influence_vec[subregion_cellnum_list])
            subregion_influence_fraction = subregion_influence / total_influence
            region_influence_summary_df = pd.concat([region_influence_summary_df, pd.DataFrame({subregion_influence_fraction})], axis=1)
        region_influence_summary_df.columns = lc_names
        influence_summary_df = pd.concat([influence_summary_df,region_influence_summary_df])
    influence_summary_df.index = region_names
    influence_summary_df.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity/influence_summary_{calculation_method}{year}{filestr}.csv', encoding='utf-8')
