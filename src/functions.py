"""
Functions used in the analysis
"""
import xarray as xr
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from glob import glob

# get start_month, end_month, campaign_name of each year
def get_campaign_info (year):
    if year == 2012:
        start_month = 5; end_month = 10; campaingn_name = 'carve'
    elif year == 2013:
        start_month = 4; end_month = 10; campaingn_name = 'carve'
    elif year == 2014:
        start_month = 5; end_month = 11; campaingn_name = 'carve'
    elif year == 2017:
        start_month = 4; end_month = 11; campaingn_name = 'arctic_cap'
    return start_month, end_month, campaingn_name


# derive the full h matrix from sparse matrix of each month
# year - year of the campaign
# n_receptor - number of receptors, same as shape[0] of h_matrix
# cell_id - a vector consisting of id of certain grids, e.g., only land grids (cell id can be found in the cell_id_table.csv)

def read_H_matrix (year, n_receptor, cell_id):
    start_month, end_month, campaign_name = get_campaign_info(year)

    for month in np.arange(start_month,end_month+1):
        print(month)

        # read stored H sparse matrix
        h_df = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/H{year}_{month}.txt",
                        sep="\s+", index_col=False, header=None,
                        names=["obs_id", "cell_id", "lat_id","lon_id", "lat", "lon", "val"])
        #  \s+ is the expression for "any amount of whitespace"

        n_cell = 720 * 120
        h_matrix0 = csr_matrix((h_df.val, (h_df.obs_id, h_df.cell_id)),  
                                shape = (n_receptor, n_cell)).toarray()
        
        h_matrix0_subset = h_matrix0[:,cell_id]

        if month == start_month:
            h_matrix = h_matrix0_subset
        else:
            h_matrix = np.concatenate((h_matrix, h_matrix0_subset), axis=1)

        del h_df, h_matrix0, h_matrix0_subset
    
    return h_matrix


# read re-gridded netcdf data and crop 30-90N region for specific year and month
def read_TRENDYv9 (data_name, var_name, year, month):

    f = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/trendy-v9/global-half-degree/{data_name}_S3_{var_name}-half-degree.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )
    ds_subset_2d = ds_subset[var_name]

    return ds_subset_2d

def read_TRENDYv11 (data_name, var_name, year, month):

    f = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/trendy-v11/global-half-degree/{data_name}_S3_{var_name}-half-degree.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )

    if var_name == 'lai':
        ds_subset_2d = ds_subset[list(ds_subset.data_vars)[0]] # ISAM LAI's varname is sometimes 'LAI' and sometimes 'lai'
    else:
        ds_subset_2d = ds_subset[var_name]

    return ds_subset_2d

def read_TRENDYv11_cPool(data_name, var_name, year):

    # read carbon pools (cLitter, cSoil) from TRENDY simulation
    if data_name == "ISBA-CTRIP":
        f = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/trendy-v11/global-half-degree/{data_name}_S3_{var_name}-half-degree-annual.nc"
    elif data_name in ['CLASSIC', 'JSBACH', 'JULES']:
        f = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/trendy-v11/global-half-degree/{data_name}_S3_{var_name}-half-degree.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year),
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )

    ds_subset_2d = ds_subset[var_name]

    return ds_subset_2d


def read_inversions (data_name, var_name, year, month):

    f = f"/central/groups/carnegie_poc/michalak-lab/data/inversions/inversions-half-degree/GCP2023/{data_name}-half-degree.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )
    ds_subset_2d = ds_subset[var_name]

    return ds_subset_2d

def read_inversions_prior (data_name, var_name, year, month):

    f = f"/central/groups/carnegie_poc/michalak-lab/data/inversions/inversions-half-degree/GCP2023-prior/{data_name}-half-degree.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )
    ds_subset_2d = ds_subset[var_name]

    return ds_subset_2d

def read_fossil_fire (data_name, var_name, year, month):

    if data_name == 'fossil':
        f = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/odiac/global-half-degree/odiac2022-half-degree-{str(year)}.nc"
    elif data_name == 'fire':
        f = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gfed/v4.1s/global-half-degree/GFED4.1s-half-degree-{str(year)}.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )
    ds_subset_2d = ds_subset[var_name]

    return ds_subset_2d

def read_remote_sensing (data_name, var_name, year, month):

    if data_name == 'par':
        f = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/ceres-par/CERES_PAR-half-degree.nc"
    elif data_name in ['fpar', 'lai']:
        f = f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/modis_fpar/{var_name}-half-degree-monthly-{year}04-{year}11.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )
    ds_subset_2d = ds_subset[var_name]

    return ds_subset_2d

def read_MODIS_VI (var_name, year, month):
    # data_name: 'ndvi', 'evi'
    # var_name: 'NDVI', 'EVI
    
    dir0 = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/modis-vi/half-degree-monthly/"
    df = pd.DataFrame({'date': [f'{year}-{str(month).zfill(2)}-01']})
    df['date'] = pd.to_datetime(df['date'])
    doystr = str(df['date'].dt.dayofyear.values[0]).zfill(3)
    filename = glob(f'MYD13C2_A{year}{doystr}*', root_dir=dir0)[0]
    f = f"{dir0}{filename}"
    
    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    lat=(ds.lat >= 30) & (ds.lat <= 90),
    )
    ds_subset_2d = ds_subset[var_name]

    return ds_subset_2d

def read_GOME2_SIF (var_name, year, month):
    # var_name: dcSIF

    f = f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/gome2/gome2_monthly_2012_2017_CF08.nc"
    
    ds = xr.open_dataset(f)
    ds = ds.where(ds != -999.)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    lat=(ds.lat >= 30) & (ds.lat <= 90),
    )
    ds_subset_2d = ds_subset[var_name]

    return ds_subset_2d

def read_cru (data_name, var_name, year, month):

    f = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/cru-jra/cru-jra-v2.3/monthly/{data_name}/{data_name}-{year}-monthly-half-degree.nc"
    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )
    ds_subset_2d = ds_subset[var_name]

    return ds_subset_2d

def read_gosif_gpp (year, month):

    f = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gosif-gpp/global-half-degree/gosif-gpp-half-degree-{year}.nc"
    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )
    ds_subset_2d = ds_subset['GOSIF_GPP'] # unit: g C m-2 mo-1

    return ds_subset_2d

def read_fluxcom_x (var_name, year, month):
    # var_nane: NEE or GPP

    f = f"/central/groups/carnegie_poc/michalak-lab/data/fluxcom-x-base/monthly-half-degree/{var_name}_{year}_050_monthly.nc"
    ds = xr.open_dataset(f)

    ds = ds.rename({'lat':'latitude', 'lon': 'longitude'})

    ds = ds.reindex(latitude=ds.latitude[::-1]) # reverse the latitude dimension

    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )
    ds_subset_2d = ds_subset[var_name] # unit gC m-2 d-1

    return ds_subset_2d


def read_abcflux (var_name, year, month):
    # var_nane: NEE, GPP, Reco

    f = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/abcflux_upscaled/half-degree/CO2Fluxes_Arctic_Boreal_{var_name}_{year}-half-degree.nc"
    ds = xr.open_dataset(f)

    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )
    ds_subset_2d = ds_subset[var_name] #unit: gC m-2 mo-1

    return ds_subset_2d