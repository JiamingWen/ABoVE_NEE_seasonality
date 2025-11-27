'''
For each month/year and each pixel, sum up the influence of all footprints (output: netcdf file)
only for selected observations
'''

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import xarray as xr

import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info


####################################################################
# sum the footprint sensitivity for selected observations
for year in [2012,2013,2014,2017]:

    start_month, end_month, campaign_name = get_campaign_info(year)
    
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
    
    for month in np.arange(start_month, end_month+1):
        print(month)
        
        # read stored H sparse matrix
        h_df = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/H{year}_{month}.txt",
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
        h_matrix_sum = h_matrix.sum(axis=0)
        h_matrix_sum_2D = h_matrix_sum.reshape((len(latitude), len(longitude)))

        ds = xr.Dataset(
            data_vars=dict(
                influence=(["latitude", "longitude"], h_matrix_sum_2D),
                ),
            coords=dict(
                longitude=(["longitude"], longitude),
                latitude=(["latitude"], latitude),
            ),
        )

        compression = dict(zlib=True, complevel=5)
        ds.to_netcdf(
            f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity/influence_sum{year}_{month}_selected.nc',
            engine="netcdf4",
            encoding={v: compression for v in ds.data_vars},
        )
        del h_matrix, h_matrix_sum, h_matrix_sum_2D, ds

    # sum over different months
    for month in np.arange(start_month, end_month+1):
        ds = xr.open_dataset(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity/influence_sum{year}_{month}_selected.nc')
        lat = ds.latitude
        lon = ds.longitude
        influence = ds.influence

        if month == start_month:
            influence_sum = influence
        else:
            influence_sum += influence

        # # plot for each month
        # fig = plt.figure(figsize=(9,6))
        # ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        # ax.gridlines()
        # ax.set_global()
        # ax.coastlines()
        # plt.title(str(month))
        # influence.plot(ax=ax, transform=ccrs.PlateCarree(),
        #         vmin=0, vmax=10, cbar_kwargs={'shrink': 0.4})
        # plt.show()

    # export influence sum nc
    compression = dict(zlib=True, complevel=5)
    influence_sum.to_netcdf(
        f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity/influence_sum{year}_selected.nc',
        engine="netcdf4",
    )