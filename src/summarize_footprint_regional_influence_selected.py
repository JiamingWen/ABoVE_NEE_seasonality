'''
For all months, sum up the 10-day influence for each observation and take the average across the observations (output: netcdf file)
only for selected observations
'''

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import os
os.chdir('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info


####################################################################
# sum the footprint sensitivity for selected observations
for year in [2012,2013,2014,2017]:

    start_month, end_month, campaign_name = get_campaign_info(year)
    output_dir = f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity'
    
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
    
    for month in np.arange(start_month, end_month+1):
        print(month)
        
        # read stored H sparse matrix
        h_df = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/monthly/H{year}_{month}.txt",
                    sep="\s+", index_col=False, header=None,
                        names=["obs_id", "cell_id", "lat_id","lon_id", "lat", "lon", "val"])
        #  \s+ is the expression for "any amount of whitespace"

        n_cell = 720 * 120
        h_matrix = csr_matrix((h_df.val, (h_df.obs_id, h_df.cell_id)),  
                                shape = (n_receptor, n_cell)).toarray()
        del h_df
        
        # only for selected observations
        h_matrix = h_matrix[mask_id,:]

        #store influence field in nc file
        longitude = np.linspace(-180.0 + 0.5 * 0.5, 180.0 - 0.5 * 0.5, round(360.0 / 0.5))
        latitude = np.linspace(30.0 + 0.5 * 0.5, 90.0 - 0.5 * 0.5, round(60.0 / 0.5))
        h_matrix_month_mean = h_matrix.mean(axis=0)
        h_matrix_month_mean_2D = h_matrix_month_mean.reshape((len(latitude), len(longitude)))

        ds = xr.Dataset(
            data_vars=dict(
                influence=(["latitude", "longitude"], h_matrix_month_mean_2D),
                ),
            coords=dict(
                longitude=(["longitude"], longitude),
                latitude=(["latitude"], latitude),
            ),
        )

        compression = dict(zlib=True, complevel=5)

        ds.to_netcdf(
            f'{output_dir}/influence_mean{year}_{month}_selected.nc',
            engine="netcdf4",
            encoding={v: compression for v in ds.data_vars},
        )
        del h_matrix_month_mean, h_matrix_month_mean_2D, ds
        
        # sum up the influence for each observation during the year
        if month == start_month:
            h_matrix_year = h_matrix
        else:
            h_matrix_year += h_matrix

    # average across the observations for the year
    h_matrix_year_mean = h_matrix_year.mean(axis=0)  # average across the observations
    #store influence field in nc file
    longitude = np.linspace(-180.0 + 0.5 * 0.5, 180.0 - 0.5 * 0.5, round(360.0 / 0.5))
    latitude = np.linspace(30.0 + 0.5 * 0.5, 90.0 - 0.5 * 0.5, round(60.0 / 0.5))
    h_matrix_year_mean_2D = h_matrix_year_mean.reshape((len(latitude), len(longitude)))

    ds_year = xr.Dataset(
        data_vars=dict(
            influence=(["latitude", "longitude"], h_matrix_year_mean_2D),
            ),
        coords=dict(
            longitude=(["longitude"], longitude),
            latitude=(["latitude"], latitude),
        ),
    )
    
    compression = dict(zlib=True, complevel=5)
    ds_year.to_netcdf(
        f'{output_dir}/influence_mean{year}_selected.nc',
        engine="netcdf4",
    )

    # # plot for each year
    # fig = plt.figure(figsize=(9,6))
    # ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    # ax.gridlines()
    # ax.set_global()
    # ax.coastlines()
    # plt.title(str(year))
    # ds_year.influence.plot(ax=ax, transform=ccrs.PlateCarree(),
    #         vmin=0, vmax=1e-2, cbar_kwargs={'shrink': 0.4})
    # plt.show()

    # concatenate h_matrix across all years
    if year == 2012:
        h_matrix_all_year = h_matrix_year
    else:
        h_matrix_all_year = np.concatenate((h_matrix_all_year, h_matrix_year), axis=0)

h_matrix_all_year_mean = h_matrix_all_year.mean(axis=0)  # average across the observations

#store influence field in nc file
longitude = np.linspace(-180.0 + 0.5 * 0.5, 180.0 - 0.5 * 0.5, round(360.0 / 0.5))
latitude = np.linspace(30.0 + 0.5 * 0.5, 90.0 - 0.5 * 0.5, round(60.0 / 0.5))
h_matrix_all_year_mean_2D = h_matrix_all_year_mean.reshape((len(latitude), len(longitude)))

ds_all_year = xr.Dataset(
    data_vars=dict(
        influence=(["latitude", "longitude"], h_matrix_all_year_mean_2D),
        ),
    coords=dict(
        longitude=(["longitude"], longitude),
        latitude=(["latitude"], latitude),
    ),
)

compression = dict(zlib=True, complevel=5)
ds_all_year.to_netcdf(
    f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/arctic_cap_airborne/h_matrix/summarized_footprint_sensitivity/influence_mean_allyears_selected.nc',
    engine="netcdf4",
    encoding={v: compression for v in ds_all_year.data_vars},
)

# # plot for all years' average
# fig = plt.figure(figsize=(9,6))
# ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
# ax.gridlines()
# ax.set_global()
# ax.coastlines()
# plt.title('All years')
# ds_all_year.plot(ax=ax, transform=ccrs.PlateCarree(),
#         vmin=0, vmax=1e-2, cbar_kwargs={'shrink': 0.4})
# plt.show()


'''plot sensitivity maps for each year'''
import cartopy.mpl.ticker as cticker
import cartopy
import cartopy.crs as ccrs
import xarray as xr
from matplotlib.colors import ListedColormap
import geopandas
def add_above_boundaries(ax):
    above_domain = geopandas.read_file(
        f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/domain/ABoVE_reference_grid_v2_1527/data/ABoVE_Study_Domain/ABoVE_Study_Domain.shp"
    )
    above_core_geometry = above_domain.loc[
        above_domain["Region"] == "Core Region", "geometry"
    ]

    globe = ccrs.Globe(
        datum="NAD83",
        ellipse="GRS80",
        semimajor_axis=6378137.0,
        semiminor_axis=6356752.314140356,
        inverse_flattening=298.257222101,
    )

    proj = ccrs.AlbersEqualArea( # the plot actually does not use it
        central_longitude=-96.0,
        central_latitude=40,
        standard_parallels=(50.0, 70.0),
        globe=globe,
    )

    ax.add_geometries(
        above_core_geometry,
        crs=proj,
        edgecolor="red",
        facecolor="none",
        zorder=1,
        linewidth=1,
        linestyle="solid"
    )

def setup_plot(ax, region_extent, ccrs_plot, axes=True):
    ax.set_extent(region_extent, ccrs_plot)
    ax.coastlines(linewidth=0.5)
    ax.set_aspect(2)

    ax.set_xticks(np.arange(-160, -80, 20), crs=ccrs_plot)
    ax.set_yticks(np.arange(50, 80, 5), crs=ccrs_plot)
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(axis='both', which='major', labelsize=14, length=8)
    if not axes:
        # ax.set_xticklabels([])
        ax.set_yticklabels([])

    return ax


region_extent = [-170, -100, 50, 75]
subtitle_loc = [-165, 72]
ccrs_plot = ccrs.PlateCarree()
fig, axs = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={'projection': ccrs_plot})
axs = axs.flatten()
for i, year in enumerate([2012, 2013, 2014, 2017]):
    campaign_name = get_campaign_info(year)[2]
    ax = axs[i]
    ax = setup_plot(ax, region_extent, ccrs_plot, axes=True)
    label = chr(97 + i)  # 'a', 'b', 'c', 'd'
    ax.text(subtitle_loc[0], subtitle_loc[1], f'({label}) {year}', fontsize=15)

    influence = xr.open_dataset(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity/influence_mean{year}_selected.nc').influence
    lons = influence["longitude"].values
    lats = influence["latitude"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cp = ax.pcolormesh(lon_grid, lat_grid, np.log10(influence), vmin=-4, vmax=-2, cmap='Purples')

    add_above_boundaries(ax)


cax = fig.add_axes([0.4, 0.01, 0.3, 0.03])
cb = fig.colorbar(cp, cax=cax, orientation="horizontal")
cb.ax.tick_params(labelsize=14)
cb.set_label(r"ppm $(\mu mol\ m^{-2}\ s^{-1})^{-1}$", fontsize=15, labelpad=10)

cb.ax.xaxis.set_label_coords(0.5, 2.5)
cb.set_ticks([-4, -3, -2])
cb.set_ticklabels([ r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'])

plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/arctic_cap_airborne/h_matrix/summarized_footprint_sensitivity/footprint_selected_by_year.png', dpi=300, bbox_inches='tight')
plt.show()