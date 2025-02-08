# temporally aggregate the 4-day tiff files into a monthly netcdf file

import glob
from typing import List
import rioxarray
import xarray as xr
import pandas as pd
import os
import numpy as np

year = 2014 # 2012 2013 2014 2017

varname = 'Fpar' #Fpar Lai
input_dir = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/modis_fpar/data_{year}'

# read filenames into a list
geotiff_list: List[str] = sorted(
    glob.glob(f"{input_dir}/MCD15A2H.061_{varname}_500m_doy*.tif")
)

# Create variable used for time axis
def extract_date(f: str):
    tmp = os.path.basename(f).split("doy")[1]
    yearstr = tmp[0:4]
    doystr = tmp[4:7]
    time_str = pd.Timestamp(f"{yearstr}-01-01") + pd.Timedelta(int(doystr)-1, 'days')
    return time_str

time_var = xr.Variable('time', [extract_date(f) for f in geotiff_list])

# Load in and concatenate all individual GeoTIFFs
geotiffs_da = xr.concat([rioxarray.open_rasterio(i) for i in geotiff_list],
                        dim=time_var)

# Covert xarray.DataArray into a xarray.Dataset
geotiffs_ds = geotiffs_da.to_dataset('band')

# Rename the variable to a more useful name
geotiffs_ds = geotiffs_ds.rename({1: varname})

# Set fill values (249-255) to be nan; range should be 0-100
geotiffs_ds = geotiffs_ds.where(geotiffs_ds[varname] <= 100)  

# Apply scaling factor 0.01 for fPAR, 0.1 for LAI
if varname == 'Fpar':
    scaling_factor = 0.01
elif varname == 'Lai':
    scaling_factor = 0.1

geotiffs_ds = geotiffs_ds * scaling_factor

# extract data
geotiffs_ds_subset: xr.Dataset = geotiffs_ds.isel(
    time=(geotiffs_ds.time.dt.year == year)
    & (geotiffs_ds.time.dt.month >= 4)
    & (geotiffs_ds.time.dt.month <= 11),
)

# Export as a NetCDF file - it is a large (1.8 GB) file, so I decide not to export it.
# compression = dict(zlib=True, complevel=5)
# geotiffs_ds_subset.to_netcdf(
#     f"/central/groups/carnegie_poc/jwen2/ABoVE/modis_fpar/{varname}-500m-8day-20170407-20171125.nc",
#     engine="netcdf4",
#     encoding={v: compression for v in geotiffs_ds_subset.data_vars},
# )

#############################################################################
# aggregate to monthly
ds_monthly = geotiffs_ds_subset.groupby("time.month").mean("time") # this takes average of all the timestamps that fall into a month, with equal weights

# # check the calculation
# geotiffs_ds_subset.Fpar[:, 6000, 16800] #(0.16 + 0.16 + 0.21)/3
# ds_monthly.Fpar[:, 6000, 16800]

# export file
compression = dict(zlib=True, complevel=5)
ds_monthly.to_netcdf(
    f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/modis_fpar/{varname}-500m-monthly-{year}04-{year}11.nc",
    engine="netcdf4",
    encoding={v: compression for v in ds_monthly.data_vars},
)

