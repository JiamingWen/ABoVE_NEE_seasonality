'''
plot footprint sensitivity map separately for daytime and nighttime
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
from scipy.sparse import csr_matrix, vstack
import os
import sys
sys.path.append('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
import utils
from functions import get_campaign_info

import cartopy.mpl.ticker as cticker
import cartopy
import cartopy.crs as ccrs
import xarray as xr
from matplotlib.colors import ListedColormap
import geopandas

# daytime: local time 9am-9pm, UTC time 6pm-6am
# night time: local time 9pm-9am, UTC time 6am-6pm
daytime_hours = [18, 21, 0, 3]
nighttime_hours = [6, 9, 12, 15]

for year in [2012, 2013, 2014, 2017]:

    print(year)

    campaign_name = get_campaign_info(year)[2]
    config = utils.getConfig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/config/config_{campaign_name}{year}_3hourly.ini')

    # read observations
    df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
    df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')
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

    h_matrix_daytime = csr_matrix((len(mask_id), 720 * 120))
    h_matrix_nighttime = csr_matrix((len(mask_id), 720 * 120))

    # read H matrix
    config = utils.getConfig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/config/config_{campaign_name}{year}_3hourly.ini')
    for ntimestep in np.arange(0, config["ntimesteps"]):
        
        print(ntimestep)
        timestep = config["sdate"] + ntimestep * config["timestep"]
        h_matrix_dir= f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/3hourly'
        h_matrix_file = f'{h_matrix_dir}/H{timestep.year}_{timestep.month}_{timestep.day}_{timestep.hour}.txt'
        
        if os.path.exists(h_matrix_file):
            # print(f"Reading {h_matrix_file}")
            h_df = pd.read_csv(
                h_matrix_file,
                sep="\s+", index_col=False, header=None,
                names=["obs_id", "cell_id", "lat_id", "lon_id", "lat", "lon", "val"]
            )
            #  \s+ is the expression for "any amount of whitespace"

            # Create sparse matrix directly
            n_cell = 720 * 120
            h_matrix0 = csr_matrix((h_df.val, (h_df.obs_id, h_df.cell_id)), shape=(n_receptor, n_cell))

        else: # no footprint falls in this time period
            h_matrix0 = csr_matrix((n_receptor, 720 * 120))

        # only for selected observations
        h_matrix0 = h_matrix0[mask_id,:]
        
        if timestep.hour in daytime_hours:
            h_matrix_daytime += h_matrix0
        elif timestep.hour in nighttime_hours:
            h_matrix_nighttime += h_matrix0

    # concatenate h_matrix across all years
    if year == 2012:
        h_matrix_daytime_all = h_matrix_daytime
        h_matrix_nighttime_all = h_matrix_nighttime
    else:
        h_matrix_daytime_all = vstack([h_matrix_daytime_all, h_matrix_daytime])
        h_matrix_nighttime_all = vstack([h_matrix_nighttime_all, h_matrix_nighttime])

h_matrix_daytime_all_mean = h_matrix_daytime_all.mean(axis=0)  # average across the observations
h_matrix_nighttime_all_mean = h_matrix_nighttime_all.mean(axis=0)

#store influence field in nc file
longitude = np.linspace(-180.0 + 0.5 * 0.5, 180.0 - 0.5 * 0.5, round(360.0 / 0.5))
latitude = np.linspace(30.0 + 0.5 * 0.5, 90.0 - 0.5 * 0.5, round(60.0 / 0.5))
h_matrix_daytime_all_mean_2D = h_matrix_daytime_all_mean.reshape((len(latitude), len(longitude)))
h_matrix_nighttime_all_mean_2D = h_matrix_nighttime_all_mean.reshape((len(latitude), len(longitude)))

compression = dict(zlib=True, complevel=5)

ds_daytime_all = xr.Dataset(
    data_vars=dict(
        influence=(["latitude", "longitude"], h_matrix_daytime_all_mean_2D),
        ),
    coords=dict(
        longitude=(["longitude"], longitude),
        latitude=(["latitude"], latitude),
    ),
)

ds_daytime_all.to_netcdf(
    f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/footprint_daytime_nightime/influence_mean_daytime_allyears_selected.nc',
    engine="netcdf4",
    encoding={v: compression for v in ds_daytime_all.data_vars},
)


ds_nighttime_all = xr.Dataset(
    data_vars=dict(
        influence=(["latitude", "longitude"], h_matrix_nighttime_all_mean_2D),
        ),
    coords=dict(
        longitude=(["longitude"], longitude),
        latitude=(["latitude"], latitude),
    ),
)

ds_nighttime_all.to_netcdf(
    f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/footprint_daytime_nightime/influence_mean_nighttime_allyears_selected.nc',
    engine="netcdf4",
    encoding={v: compression for v in ds_nighttime_all.data_vars},
)



'''plot spatial maps'''
def add_above_boundaries(ax):
    above_domain = geopandas.read_file(
        f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/domain/ABoVE_reference_grid_v2_1527/data/ABoVE_Study_Domain/ABoVE_Study_Domain.shp"
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

dir = '/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/footprint_daytime_nightime'
region_extent = [-170, -100, 50, 75]
subtitle_loc = [-165, 72]
ccrs_plot = ccrs.PlateCarree()
fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), subplot_kw={'projection': ccrs_plot})
for i, time_of_day in enumerate(['daytime', 'nighttime']):

    ax = axs[i]
    ax = setup_plot(ax, region_extent, ccrs_plot, axes=True)
    label = 'a' if i == 0 else 'b'
    ax.text(subtitle_loc[0], subtitle_loc[1], f'({label}) {time_of_day.capitalize()}', fontsize=15)

    influence = xr.open_dataset(f'{dir}/influence_mean_{time_of_day}_allyears_selected.nc').influence
    lons = influence["longitude"].values
    lats = influence["latitude"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cp = ax.pcolormesh(lon_grid, lat_grid, np.log10(influence), vmin=-4, vmax=-2, cmap='Purples')
    add_above_boundaries(ax)

cax = fig.add_axes([0.4, 0.01, 0.3, 0.03])
cb = fig.colorbar(cp, cax=cax, orientation="horizontal")
cb.ax.tick_params(labelsize=14)
cb.set_label(r"ppm $(\mu mol\ m^{-2}\ s^{-1})^{-1}$", fontsize=15, labelpad=10)

cb.ax.xaxis.set_label_coords(0.5, 3.5)
cb.set_ticks([-4, -3, -2])
cb.set_ticklabels([ r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'])

plt.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/footprint_daytime_nightime/footprint_daytime_nighttime_map.png', dpi=300, bbox_inches='tight')
plt.show()