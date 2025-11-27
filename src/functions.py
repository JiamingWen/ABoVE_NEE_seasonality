"""
Functions used in the analysis
"""
import xarray as xr
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
import matplotlib.cm as cm
import os
import datetime
import utils

# get start_month, end_month, campaign_name of each year
def get_campaign_info (year):
    if year == 2012:
        start_month = 5; end_month = 10; campaingn_name = 'carve'
    elif year == 2013:
        start_month = 3; end_month = 10; campaingn_name = 'carve'
    elif year == 2014:
        start_month = 4; end_month = 11; campaingn_name = 'carve'
    elif year == 2017:
        start_month = 4; end_month = 11; campaingn_name = 'arctic_cap'
    return start_month, end_month, campaingn_name


# derive the full h matrix from sparse matrix of each month
# year - year of the campaign
# n_receptor - number of receptors, same as shape[0] of h_matrix
# cell_id - a vector consisting of id of certain grids, e.g., only land grids (cell id can be found in the cell_id_table.csv)

def read_H_matrix(year, n_receptor, cell_id):
    start_month, end_month, campaign_name = get_campaign_info(year)

    h_matrix = None

    for month in np.arange(start_month, end_month + 1):
        print(month)

        # read stored H sparse matrix
        h_df = pd.read_csv(
            f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/monthly/H{year}_{month}.txt",
            sep="\s+", index_col=False, header=None,
            names=["obs_id", "cell_id", "lat_id", "lon_id", "lat", "lon", "val"]
        )
        #  \s+ is the expression for "any amount of whitespace"

        # Create sparse matrix directly
        n_cell = 720 * 120
        h_matrix0 = csr_matrix((h_df.val, (h_df.obs_id, h_df.cell_id)), shape=(n_receptor, n_cell))

        # Subset the sparse matrix
        h_matrix0_subset = h_matrix0[:, cell_id]

        # Concatenate sparse matrices
        if h_matrix is None:
            h_matrix = h_matrix0_subset
        else:
            h_matrix = hstack([h_matrix, h_matrix0_subset]).tocsr()

        del h_df, h_matrix0, h_matrix0_subset

    return h_matrix

def read_H_matrix_3hourly(year, n_receptor, cell_id):

    # derive the full h matrix from sparse matrix of each 3-hourly timestep
    # year - year of the campaign
    # n_receptor - number of receptors, same as shape[0] of h_matrix
    # cell_id - a vector consisting of id of certain grids, e.g., only land grids (cell id can be found in the cell_id_table.csv)

    campaign_name = get_campaign_info(year)[2]
    config = utils.getConfig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/config/config_{campaign_name}{year}_3hourly.ini')
    
    h_matrix = None
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

        # Subset the sparse matrix
        h_matrix0_subset = h_matrix0[:, cell_id]

        # Concatenate sparse matrices
        if h_matrix is None:
            h_matrix = h_matrix0_subset
        else:
            h_matrix = hstack([h_matrix, h_matrix0_subset]).tocsr()

        del h_matrix0, h_matrix0_subset

    return h_matrix


def read_H_matrix_daily(year, n_receptor, cell_id):

    # derive the full h matrix from sparse matrix of each daily timestep
    # year - year of the campaign
    # n_receptor - number of receptors, same as shape[0] of h_matrix
    # cell_id - a vector consisting of id of certain grids, e.g., only land grids (cell id can be found in the cell_id_table.csv)

    campaign_name = get_campaign_info(year)[2]
    config = utils.getConfig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/config/config_{campaign_name}{year}_3hourly.ini')

    h_matrix = None
    for nday in np.arange(0, config["ndays"]):
        
        print(nday)
        timestep = config["sdate"] + nday * datetime.timedelta(days=1)
        h_matrix_dir= f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/daily'
        h_matrix_file = f'{h_matrix_dir}/H{timestep.year}_{timestep.month}_{timestep.day}.txt'
        
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

        # Subset the sparse matrix
        h_matrix0_subset = h_matrix0[:, cell_id]

        # Concatenate sparse matrices
        if h_matrix is None:
            h_matrix = h_matrix0_subset
        else:
            h_matrix = hstack([h_matrix, h_matrix0_subset]).tocsr()

        del h_matrix0, h_matrix0_subset

    return h_matrix


def read_H_matrix_monthly(year, n_receptor, cell_id):

    # derive the full h matrix from sparse matrix of each monthly timestep
    # year - year of the campaign
    # n_receptor - number of receptors, same as shape[0] of h_matrix
    # cell_id - a vector consisting of id of certain grids, e.g., only land grids (cell id can be found in the cell_id_table.csv)

    campaign_name = get_campaign_info(year)[2]
    config = utils.getConfig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/config/config_{campaign_name}{year}_3hourly.ini')

    start_month = config["sdate"].month
    end_month = config["edate"].month

    h_matrix = None
    for month in np.arange(start_month, end_month+1):
        
        print(month)
        h_matrix_dir= f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/monthly'
        h_matrix_file = f'{h_matrix_dir}/H{year}_{month}.txt'
        
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

        # Subset the sparse matrix
        h_matrix0_subset = h_matrix0[:, cell_id]

        # Concatenate sparse matrices
        if h_matrix is None:
            h_matrix = h_matrix0_subset
        else:
            h_matrix = hstack([h_matrix, h_matrix0_subset]).tocsr()

        del h_matrix0, h_matrix0_subset

    return h_matrix


# read re-gridded netcdf data for specific year and month, rename latitude and longitude if needed
def read_TRENDYv9 (data_name, var_name, year, month):

    f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/trendy-v9/global-half-degree/{data_name}_S3_{var_name}-half-degree.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )
    ds_subset_variable = ds_subset[var_name]

    return ds_subset_variable

def read_TRENDYv11 (data_name, var_name, year, month):

    f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/trendy-v11/global-half-degree/{data_name}_S3_{var_name}-half-degree.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )

    if var_name == 'lai':
        ds_subset_variable = ds_subset[list(ds_subset.data_vars)[0]] # ISAM LAI's varname is sometimes 'LAI' and sometimes 'lai'
    else:
        ds_subset_variable = ds_subset[var_name]

    return ds_subset_variable

def read_TRENDYv11_cPool(data_name, var_name, year):

    # read carbon pools (cLitter, cSoil) from TRENDY simulation
    if data_name == "ISBA-CTRIP":
        f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/trendy-v11/global-half-degree/{data_name}_S3_{var_name}-half-degree-annual.nc"
    elif data_name in ['CLASSIC', 'JSBACH', 'JULES']:
        f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/trendy-v11/global-half-degree/{data_name}_S3_{var_name}-half-degree.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year),
    )

    ds_subset_variable = ds_subset[var_name]

    return ds_subset_variable


def read_inversions (data_name, var_name, year, month):

    f = f"/resnick/groups/carnegie_poc/michalak-lab/data/inversions/inversions-half-degree/GCP2023/{data_name}-half-degree.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )
    ds_subset_variable = ds_subset[var_name]

    return ds_subset_variable

def read_inversions_prior (data_name, var_name, year, month):

    f = f"/resnick/groups/carnegie_poc/michalak-lab/data/inversions/inversions-half-degree/GCP2023-prior/{data_name}-half-degree.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )
    ds_subset_variable = ds_subset[var_name]

    return ds_subset_variable

def read_fossil (data_name, year, month):

    if data_name == 'odiac2022':
        f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/odiac/global-half-degree/odiac2022-half-degree-{str(year)}.nc"
        var_name = 'land'
    elif data_name == 'gridfed2024':
        f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gcp-gridfed/global-half-degree/GCP-GridFEDv2024.0_{year}-half-degree.nc"
        var_name = 'TOTAL'
    else:
        raise ValueError("data_name is not valid")

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )
    ds_subset_variable = ds_subset[var_name]

    return ds_subset_variable

def read_fire (data_name, year, month):

    if data_name == 'gfed4.1':
        f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gfed/v4.1s/global-half-degree/GFED4.1s-half-degree-{year}.nc"
        var_name = 'CO2_emission'
    elif data_name == 'gfed5':
        f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gfed/v5/global-half-degree/GFED5_Beta_monthly-half-degree-{year}.nc"
        var_name = 'CO2'

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )
    ds_subset_variable = ds_subset[var_name]

    return ds_subset_variable

def read_remote_sensing (data_name, var_name, year, month):

    if data_name == 'par':
        f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/ceres-par/CERES_PAR-half-degree.nc"
    elif data_name in ['fpar', 'lai']:
        f = f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/modis_fpar/{var_name}-half-degree-monthly-{year}04-{year}11.nc"

    ds = xr.open_dataset(f)

    if data_name in ['fpar', 'lai'] and month == 3: # I did not download data for month 3 - just set the values as 0, it should have little effect on the results
        ds_subset: xr.Dataset = ds.isel(
        time=(ds.time.dt.year == year) & (ds.time.dt.month == 4),
        )
        ds_subset[var_name].data[:] = np.nan
    else:
        ds_subset: xr.Dataset = ds.isel(
        time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
        )

    ds_subset_variable = ds_subset[var_name]

    return ds_subset_variable

def read_MODIS_VI (var_name, year, month):
    # data_name: 'ndvi', 'evi'
    # var_name: 'NDVI', 'EVI
    
    dir0 = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/modis-vi/half-degree-monthly/"
    df = pd.DataFrame({'date': [f'{year}-{str(month).zfill(2)}-01']})
    df['date'] = pd.to_datetime(df['date'])
    doystr = str(df['date'].dt.dayofyear.values[0]).zfill(3)
    filename = glob(f'MYD13C2_A{year}{doystr}*', root_dir=dir0)[0]
    f = f"{dir0}{filename}"
    
    ds = xr.open_dataset(f)
    ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )

    ds_subset_variable = ds_subset[var_name]

    return ds_subset_variable

def read_GOME2_SIF (var_name, year, month):
    # var_name: dcSIF

    f = f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/gome2/gome2_monthly_2012_2017_CF08.nc"
    
    ds = xr.open_dataset(f)
    ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    ds = ds.where(ds != -999.)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )
    ds_subset_variable = ds_subset[var_name]

    return ds_subset_variable

def read_cru (data_name, var_name, year, month):

    f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/cru-jra/cru-jra-v2.3/monthly/{data_name}/{data_name}-{year}-monthly-half-degree.nc"
    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )
    ds_subset_variable = ds_subset[var_name]

    return ds_subset_variable

def read_gosif_gpp (year, month):

    f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gosif-gpp/global-half-degree/gosif-gpp-half-degree-{year}.nc"
    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )
    ds_subset_variable = ds_subset['GOSIF_GPP'] # unit: g C m-2 mo-1

    return ds_subset_variable

def read_x_base_monthly (var_name, year, month):
    # var_name: NEE or GPP

    f = f"/resnick/groups/carnegie_poc/michalak-lab/data/fluxcom-x-base/monthly-half-degree/{var_name}_{year}_050_monthly.nc"
    ds = xr.open_dataset(f)

    ds = ds.rename({'lat':'latitude', 'lon': 'longitude'})

    ds = ds.reindex(latitude=ds.latitude[::-1]) # reverse the latitude dimension

    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )
    ds_subset_variable = ds_subset[var_name] # unit gC m-2 d-1

    return ds_subset_variable

def read_x_base_daily(year, month, date):
    dir0 = '/resnick/groups/carnegie_poc/michalak-lab/data/fluxcom-x-base/daily/'
    filename = f"{dir0}/NEE_{year}_05_daily.nc"
    ds = xr.open_dataset(filename, decode_coords="all") #unit: gC m-2 d-1
    ds_daily = ds.sel(time=f"{year}-{month:02d}-{date:02d}T00:00:00", method="nearest")
    ds_daily = ds_daily.drop_vars("time", errors="ignore")
    ds_daily = ds_daily['NEE']

    return ds_daily

def read_x_base_monthlycycle(year: int, month: int) -> xr.DataArray:
    '''read monthly diurnal cycle'''
    dir_monthlycycle: str = "/resnick/groups/carnegie_poc/michalak-lab/data/fluxcom-x-base/monthlycycle"
    f = f"{dir_monthlycycle}/NEE_{year}_05_monthlycycle.nc"
    ds_monthlycycle = xr.open_dataset(f, decode_coords=None) #unit: gC m-2 d-1
    ds_monthlycycle = ds_monthlycycle.sel(time=f"{year}-{month:02d}-01T00:00:00", method="nearest")
    return ds_monthlycycle

def read_abcflux (var_name, year, month):
    # var_name: NEE, GPP, Reco

    f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/abcflux_upscaled/half-degree/CO2Fluxes_Arctic_Boreal_{var_name}_{year}-half-degree.nc"
    ds = xr.open_dataset(f)

    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )
    ds_subset_variable = ds_subset[var_name] #unit: gC m-2 mo-1

    return ds_subset_variable

def read_ocean_fluxes(data_name, year, month):

    f = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gcb-2024-ocean/global-half-degree/GCB-2024_dataprod_{data_name}_2000-2023-half-degree.nc"

    ds = xr.open_dataset(f)
    ds_subset: xr.Dataset = ds.isel(
    time=(ds.time.dt.year == year) & (ds.time.dt.month == month),
    )
    ds_subset_variable = ds_subset['fgco2'] #unit: mol m-2 s-1

    return ds_subset_variable


def read_CT_NOAA_3hourly(year, month, date, hour):
    dir0 = '/resnick/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/CT-NOAA/three-hourly/'
    filename = f'{dir0}CT2022.flux0.5x0.5.{year}{month:02d}{date:02d}.nc'
    ds = xr.open_dataset(filename)

    ds_subset: xr.Dataset = ds.isel(
    time=hour // 3,
    )
    ds_subset_2d = ds_subset['bio_flux_opt']

    return ds_subset_2d


def read_CT_NOAA_daily(year, month, date):
    dir0 = '/resnick/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/CT-NOAA/daily/'
    filename = f'{dir0}CT2022.flux0.5x0.5.{year}{month:02d}{date:02d}.nc'
    ds = xr.open_dataset(filename)

    ds_subset: xr.Dataset = ds.isel(
    )
    ds_subset_2d = ds_subset['bio_flux_opt']

    return ds_subset_2d

def read_CT_NOAA_monthly(year, month):
    dir0 = '/resnick/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/CT-NOAA/monthly/'
    filename = f'{dir0}CT2022.flux0.5x0.5.{year}{month:02d}.nc'
    ds = xr.open_dataset(filename)

    ds_subset: xr.Dataset = ds.isel(
    )
    ds_subset_2d = ds_subset['bio_flux_opt']

    return ds_subset_2d

def read_CTE_3hourly(year, month, date, hour):
    dir0 = '/resnick/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/CTE/three-hourly/'
    filename = f'{dir0}bio_{year}{month:02d}{date:02d}.nc'
    ds = xr.open_dataset(filename)

    ds_subset: xr.Dataset = ds.isel(
    time=hour // 3,
    )
    ds_subset_2d = ds_subset['co2_bio_flux_opt']

    return ds_subset_2d

def read_CTE_daily(year, month, date):
    dir0 = '/resnick/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/CTE/daily/'
    filename = f'{dir0}bio_{year}{month:02d}{date:02d}.nc'
    ds = xr.open_dataset(filename)

    ds_subset: xr.Dataset = ds.isel(
    )
    ds_subset_2d = ds_subset['co2_bio_flux_opt']

    return ds_subset_2d

def read_CTE_monthly(year, month):
    dir0 = '/resnick/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/CTE/monthly/'
    filename = f'{dir0}bio_{year}{month:02d}.nc'
    ds = xr.open_dataset(filename)

    ds_subset: xr.Dataset = ds.isel(
    )
    ds_subset_2d = ds_subset['co2_bio_flux_opt']

    return ds_subset_2d

def subset_30N_90N (ds):
    # crop 30-90N region to match the footprints of airborne observations
    ds_subset: xr.Dataset = ds.isel(
    latitude=(ds.latitude >= 30) & (ds.latitude <= 90),
    )

    return ds_subset