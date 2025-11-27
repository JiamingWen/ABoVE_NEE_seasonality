'''check X-BASE outputs at different temporal resolutions'''

import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

year = 2012
month = 7
lat0 = 42.25
lon0 = -76.25

dir_monthlydiurnal = '/central/groups/carnegie_poc/michalak-lab/data/fluxcom-x-base/monthlycycle'
dir_daily = '/central/groups/carnegie_poc/michalak-lab/data/fluxcom-x-base/daily'
dir_monthly = '/central/groups/carnegie_poc/michalak-lab/data/fluxcom-x-base/monthly-half-degree'

# monthly 0.5 degree
filename = f"{dir_monthly}/NEE_{year}_050_monthly.nc"
ds = xr.open_dataset(filename, decode_coords="all") #unit: gC m-2 d-1
ds_monthly050 = ds['NEE'].isel(time=(ds.time.dt.year == year) & (ds.time.dt.month == month))
ds_monthly050_value = ds_monthly050.sel(lat=lat0, lon=lon0, method="nearest").values
print(ds_monthly050_value)
# 0.8403384

# daily 0.5 degree
filename = f"{dir_daily}/NEE_{year}_05_daily.nc"
ds = xr.open_dataset(filename, decode_coords="all") #unit: gC m-2 d-1
ds_daily050 = ds['NEE'].isel(time=(ds.time.dt.year == year) & (ds.time.dt.month == month))
ds_daily050_value = ds_daily050.sel(latitude=lat0, longitude=lon0, method="nearest").values
print(np.mean(ds_daily050_value))
# 0.8402859

# monthly diurnal cycle 0.5 degree
filename = f"{dir_monthlydiurnal}/NEE_{year}_05_monthlycycle.nc"
ds = xr.open_dataset(filename, decode_coords="all") #unit: gC m-2 d-1
ds_monthlycycle050 = ds['NEE'].isel(time=(ds.time.dt.year == year) & (ds.time.dt.month == month))
ds_monthlycycle050_value = ds_monthlycycle050.sel(latitude=lat0, longitude=lon0, method="nearest").values
print(np.mean(ds_monthlycycle050_value))
# 0.84028596

# the last two values are exactly the same, and have slight difference (~1e-5) with the first value likely due to regridding precision