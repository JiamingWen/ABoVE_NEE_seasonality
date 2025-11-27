'''
For each footprint, sum up influence from different land covers (output: csv file)
For each month/year and each pixel, sum up influence to all footprints (output: netcdf file)
'''

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import xarray as xr

import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info

cell_id_table = pd.read_csv('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/cell_id_table/cell_id_table.csv')
ABoVEcore_cellnum_list = np.where(cell_id_table['ABoVE'] == 0)[0]
ABoVE_cellnum_list = np.where(cell_id_table['ABoVE'].isin ([0,1]))[0]
ocean_cellnum_list = np.where(cell_id_table['land'] == 0)[0]
forest_cellnum_list = np.where(cell_id_table['lc'] == 5)[0]
shrub_cellnum_list = np.where(cell_id_table['lc'] == 7)[0]
tundra_cellnum_list = np.where(cell_id_table['lc'].isin([8,9,10]))[0]


for year in [2012, 2013, 2014, 2017]:

    start_month, end_month, campaign_name = get_campaign_info(year)
    output_dir = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    receptor_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
    n_receptor = receptor_df.shape[0]
    del receptor_df

    # sum up influence from H matrix
    result_df = pd.DataFrame()

    for month in np.arange(start_month, end_month+1):
        print(month)
        
        # read stored H sparse matrix
        h_df = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/monthly/H{year}_{month}.txt",
                        sep="\s+", index_col=False, header=None,
                        names=["obs_id", "cell_id", "lat_id","lon_id", "lat", "lon", "val"])
        #  \s+ is the expression for "any amount of whitespace"

        n_cell = 720 * 120
        h_matrix = csr_matrix((h_df.val, (h_df.obs_id, h_df.cell_id)),  
                                shape = (n_receptor, n_cell)).toarray()
        del h_df


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
            f'{output_dir}influence_mean{year}_{month}.nc',
            engine="netcdf4",
            encoding={v: compression for v in ds.data_vars},
        )
        del h_matrix_month_mean, h_matrix_month_mean_2D, ds


        # sum up the influence (i.e., 10-day cumulative) for each observation during the year
        if month == start_month:
            h_matrix_year = h_matrix
        else:
            h_matrix_year += h_matrix


        #summarize the influence from the ABoVE region for each observation
        # I am using H matrix itself now - maybe consider times fluxes from ocean and land?
        result_df['total_influence'] = h_matrix_year.sum(axis=1)
        result_df['ABoVEcore_influence'] = h_matrix_year[:,ABoVEcore_cellnum_list].sum(axis=1)
        result_df['ABoVE_influence'] = h_matrix_year[:,ABoVE_cellnum_list].sum(axis=1)
        result_df['ocean_influence'] = h_matrix_year[:,ocean_cellnum_list].sum(axis=1)
        tmp = np.delete(h_matrix_year, ocean_cellnum_list, axis=1)
        result_df['land_influence'] = tmp.sum(axis=1); del tmp
        result_df['forest_influence'] = h_matrix_year[:,forest_cellnum_list].sum(axis=1)
        result_df['shrub_influence'] = h_matrix_year[:,shrub_cellnum_list].sum(axis=1)
        result_df['tundra_influence'] = h_matrix_year[:,tundra_cellnum_list].sum(axis=1)

        result_df['ABoVE_influence_fraction'] = result_df['ABoVE_influence'] / result_df['total_influence']
        result_df['ABoVE_land_influence_fraction'] = result_df['ABoVE_influence'] / result_df['land_influence']
        result_df['ocean_influence_fraction'] = result_df['ocean_influence'] / result_df['total_influence']
        result_df.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv', encoding='utf-8', index=False)

    # average across observations for the year
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
        f'{output_dir}/influence_mean{year}.nc',
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
    f'{output_dir}influence_mean_allyears.nc',
    engine="netcdf4",
    encoding={v: compression for v in ds_all_year.data_vars},
)