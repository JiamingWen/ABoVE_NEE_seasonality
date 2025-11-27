# compare CARVE and Arctic-CAP hourly footprints

import numpy as np
import pandas as pd
import xarray as xr
import time


def read_monthlycycle_x_base(year: int, month: int) -> xr.DataArray:
    '''read monthly diurnal cycle'''
    dir_monthlycycle: str = "/central/groups/carnegie_poc/michalak-lab/data/fluxcom-x-base/monthlycycle"
    f = f"{dir_monthlycycle}/NEE_{year}_05_monthlycycle.nc"
    ds_monthlycycle = xr.open_dataset(f, decode_coords=None) #unit: gC m-2 d-1
    ds_monthlycycle = ds_monthlycycle.sel(time=f"{year}-{month:02d}-01T00:00:00", method="nearest")
    return ds_monthlycycle

def read_daily_x_base(year: int, month: int, day: int) -> xr.DataArray:
    """Read daily X-BASE data"""
    dir_daily: str = "/central/groups/carnegie_poc/michalak-lab/data/fluxcom-x-base/daily"
    f = f"{dir_daily}/NEE_{year}_05_daily.nc"
    ds = xr.open_dataset(f, decode_coords="all") #unit: gC m-2 d-1
    ds_daily = ds.sel(time=f"{year}-{month:02d}-{day:02d}T00:00:00", method="nearest")
    ds_daily = ds_daily.drop_vars("time", errors="ignore")
    return ds_daily


def reorganize_xarray (ds):
    # reorganize lat/lon to match with H matrix of footprint

    # original:
    # latitude: -89.75, ..., 89.75
    # longitude: -179.75, ..., 179.75

    # target:
    # latitude: 30.25, ..., 89.75
    # longitude: 140.25, ..., 179.75, -179.75, ..., 139.75

    ds_subset = ds.sel(latitude=slice(30.25, 89.75))

    ds_subset = xr.concat(
        [ds_subset.sel(longitude=slice(140.25, 179.75)), 
         ds_subset.sel(longitude=slice(-179.75, 139.75))], 
        dim="longitude"
    )

    return ds_subset


_OCEAN_MASK = xr.open_dataset(
    '/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/above_mask/ocean-mask-half-degree.nc'
)
_OCEAN_MASK = _OCEAN_MASK.assign_coords(
    longitude=((( _OCEAN_MASK.longitude + 180) % 360) - 180)
).sortby('longitude')

def mask_ocean_pixels (ds):
    # only select land pixels - this may lead to small differences in CO2 enhancement (e.g., 0.08 ppm)
    return ds.where(_OCEAN_MASK['seamask'] == 0)

'''2017'''
year = 2017 # 2012 2013 2014 2017
if year in [2012, 2013, 2014]:
    campaign_name = 'carve'
    dir_footprint = f'/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/carve/CARVE_L4_WRF-STILT_Footprint/data/CARVE-{year}-aircraft-footprints-convect/'
else:
    campaign_name = 'arctic_cap'
    dir_footprint = '/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/above/ABoVE_Footprints_WRF_AK_NWCa/data/ArcticCAP_2017_insitu-footprints/'

# read atmospheric observations
df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

# filters for airborne observations
mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
    (df_influence['ABoVE_influence_fraction'] > 0.5) &
    (df_influence['ocean_influence_fraction'] < 0.3) &
    (df_airborne['CO2_change'] < 30) &
    (df_airborne['CO_change'] < 40))[0]

row_id = 29
start_time = time.time()
print(f"Processing row {row_id} of {len(df_airborne)}")
footprint_filename = df_airborne.at[row_id, 'footprint_filename']
footprint_ds = xr.open_dataset(filename_or_obj=dir_footprint+footprint_filename)
foot1lat = footprint_ds.foot1lat
foot1lon = footprint_ds.foot1lon
foot1date = footprint_ds.foot1date
foot1 = footprint_ds.foot1
foot1 = footprint_ds.foot1.rename({'foot1lat': 'latitude', 'foot1lon': 'longitude'})

CO2_change = 0.0
month0 = -1; day0 = -1
for timestamp in foot1date.values:
    print(timestamp)
    start_time1 = time.time()

    timestamp_str = str(timestamp)
    year = int(timestamp_str[:4])
    month = int(timestamp_str[5:7])
    day = int(timestamp_str[8:10])
    hour = int(timestamp_str[11:13])

    # footprint
    foot1_hourly = foot1.sel(foot1date=timestamp)
    foot1_hourly = foot1_hourly.drop_vars("foot1date", errors="ignore")
    w = np.ascontiguousarray(foot1_hourly.data, dtype=np.float32)
    print("footprint to np --- %s seconds ---" % (time.time() - start_time1)) # 2 sec

    '''hourly X-BASE'''
    if day != day0: # if it is on the same day, do not need to read the data again, but this does not save time, the main time consuming part is 
        ds_daily = read_daily_x_base(year, month, day)
        ds_daily = reorganize_xarray(ds_daily)
        ds_daily = mask_ocean_pixels(ds_daily)
        ds_daily = ds_daily.fillna(0)
        print("read daily --- %s seconds ---" % (time.time() - start_time1)) # 14 sec, 8 sec if changing the unit later
        nee_daily = np.ascontiguousarray(ds_daily['NEE'].data, dtype=np.float32) 
        print("daily to np --- %s seconds ---" % (time.time() - start_time1))

    ds_daily_t = np.tensordot(nee_daily, w, axes=((0,1),(0,1))) # this step somehow takes much longer time for 2017 Arctic-CAP data than 2012 CARVE data
    # ds_daily_t = xr.dot(ds_daily['NEE'], foot1_hourly, dims=["latitude", "longitude"])
    print("daily transported --- %s seconds ---" % (time.time() - start_time1))

    if month != month0: # if it is in the same month, do not need to read the data again
        ds_monthlycycle = read_monthlycycle_x_base(year, month)
        ds_monthlycycle = reorganize_xarray(ds_monthlycycle)
        ds_monthlycycle = mask_ocean_pixels(ds_monthlycycle)
        ds_monthlycycle = ds_monthlycycle.fillna(0)
        # print("read monthly diurnal cycle --- %s seconds ---" % (time.time() - start_time1))

    ds_hourly_deviation = ds_monthlycycle.sel(hour=hour, method="nearest")
    nee_hourly_deviation = np.ascontiguousarray(ds_hourly_deviation['NEE'].data, dtype=np.float32)
    ds_hourly_deviation_t = np.tensordot(nee_hourly_deviation, w, axes=((0,1),(0,1)))

    # ds_hourly_deviation = ds_hourly_deviation.drop_vars("time", errors="ignore")
    # ds_hourly_deviation = ds_hourly_deviation.drop_vars("hour", errors="ignore")
    # ds_hourly_deviation_t = xr.dot(ds_hourly_deviation['NEE'], foot1_hourly, dims=["latitude", "longitude"])
    print("monthly diurnal cycle to np --- %s seconds ---" % (time.time() - start_time1))

    tmp = ds_daily_t + ds_hourly_deviation_t # I decided to separately apply footprint to daily and monthly cycle, then add them together; it takes longer time if I add the two xarray first
    CO2_change += tmp/24/3600/12*1e6 # convert unit from gC m-2 d-1 to μmol m-2 s-1

    # save time info of this timestamp
    month0 = month
    day0 = day

    print("--- %s seconds ---" % (time.time() - start_time1)) # 14 sec or 2 sec, depending on whether it reads daily X-BASE

print("--- %s seconds ---" % (time.time() - start_time)) # 684 sec, 319 sec if changing unit later

# save some important inputs
footprint_filename_2017 = footprint_filename
foot1_2017 = foot1



'''2012'''
year = 2012 # 2012 2013 2014 2017
if year in [2012, 2013, 2014]:
    campaign_name = 'carve'
    dir_footprint = f'/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/carve/CARVE_L4_WRF-STILT_Footprint/data/CARVE-{year}-aircraft-footprints-convect/'
else:
    campaign_name = 'arctic_cap'
    dir_footprint = '/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/above/ABoVE_Footprints_WRF_AK_NWCa/data/ArcticCAP_2017_insitu-footprints/'

# read atmospheric observations
df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

# filters for airborne observations
mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
    (df_influence['ABoVE_influence_fraction'] > 0.5) &
    (df_influence['ocean_influence_fraction'] < 0.3) &
    (df_airborne['CO2_change'] < 30) &
    (df_airborne['CO_change'] < 40))[0]

row_id = 0
start_time = time.time()
print(f"Processing row {row_id} of {len(df_airborne)}")
footprint_filename = df_airborne.at[row_id, 'footprint_filename']
footprint_ds = xr.open_dataset(filename_or_obj=dir_footprint+footprint_filename)
foot1lat = footprint_ds.foot1lat
foot1lon = footprint_ds.foot1lon
foot1date = footprint_ds.foot1date
foot1 = footprint_ds.foot1
foot1 = footprint_ds.foot1.rename({'foot1lat': 'latitude', 'foot1lon': 'longitude'})

CO2_change = 0.0
month0 = -1; day0 = -1
for timestamp in foot1date.values:
    print(timestamp)
    start_time1 = time.time()

    timestamp_str = str(timestamp)
    year = int(timestamp_str[:4])
    month = int(timestamp_str[5:7])
    day = int(timestamp_str[8:10])
    hour = int(timestamp_str[11:13])

    # footprint
    foot1_hourly = foot1.sel(foot1date=timestamp, method="nearest")
    foot1_hourly = foot1_hourly.drop_vars("foot1date", errors="ignore")
    w = np.ascontiguousarray(foot1_hourly.data, dtype=np.float32)
    print("footprint to np --- %s seconds ---" % (time.time() - start_time1)) # 0.09

    '''hourly X-BASE'''
    if day != day0: # if it is on the same day, do not need to read the data again, but this does not save time, the main time consuming part is 
        ds_daily = read_daily_x_base(year, month, day)
        ds_daily = reorganize_xarray(ds_daily)
        ds_daily = mask_ocean_pixels(ds_daily)
        ds_daily = ds_daily.fillna(0)
        print("read daily --- %s seconds ---" % (time.time() - start_time1)) # 14 sec
        nee_daily = np.ascontiguousarray(ds_daily['NEE'].data, dtype=np.float32) 
        print("daily to np --- %s seconds ---" % (time.time() - start_time1))

    ds_daily_t = np.tensordot(nee_daily, w, axes=((0,1),(0,1)))
    # ds_daily_t = xr.dot(ds_daily['NEE'], foot1_hourly, dims=["latitude", "longitude"])
    print("daily to np --- %s seconds ---" % (time.time() - start_time1))

    if month != month0: # if it is in the same month, do not need to read the data again
        ds_monthlycycle = read_monthlycycle_x_base(year, month)
        ds_monthlycycle = reorganize_xarray(ds_monthlycycle)
        ds_monthlycycle = mask_ocean_pixels(ds_monthlycycle)
        ds_monthlycycle = ds_monthlycycle.fillna(0)

    ds_hourly_deviation = ds_monthlycycle.sel(hour=hour, method="nearest")
    nee_hourly_deviation = np.ascontiguousarray(ds_hourly_deviation['NEE'].data, dtype=np.float32)
    ds_hourly_deviation_t = np.tensordot(nee_hourly_deviation, w, axes=((0,1),(0,1)))

    # ds_hourly_deviation = ds_hourly_deviation.drop_vars("time", errors="ignore")
    # ds_hourly_deviation = ds_hourly_deviation.drop_vars("hour", errors="ignore")
    # ds_hourly_deviation_t = xr.dot(ds_hourly_deviation['NEE'], foot1_hourly, dims=["latitude", "longitude"])
    print("monthly diurnal cycle to np --- %s seconds ---" % (time.time() - start_time1))

    tmp = ds_daily_t + ds_hourly_deviation_t # I decided to separately apply footprint to daily and monthly cycle, then add them together; it takes longer time if I add the two xarray first
    CO2_change += tmp/24/3600/12*1e6 # convert unit from gC m-2 d-1 to μmol m-2 s-1

    # save time info of this timestamp
    month0 = month
    day0 = day

    print("--- %s seconds ---" % (time.time() - start_time1)) 

print("--- %s seconds ---" % (time.time() - start_time)) # 185 sec


# save some important inputs
footprint_filename_2012 = footprint_filename
foot1_2012 = foot1



'''compare footprint -> np calculation between 2012 and 2017'''
foot1 = foot1_2017 #foot1_2017 foot1_2012
id = 80

timestamp = foot1date.values[10]
start_time1 = time.time()
timestamp_str = str(timestamp)
year = int(timestamp_str[:4])
month = int(timestamp_str[5:7])
day = int(timestamp_str[8:10])
hour = int(timestamp_str[11:13])

# footprint
foot1_hourly = foot1.sel(foot1date=timestamp, method="nearest")
print("footprint to np --- %s seconds ---" % (time.time() - start_time1)) # 0.09
foot1_hourly = foot1_hourly.drop_vars("foot1date", errors="ignore")
print("footprint to np --- %s seconds ---" % (time.time() - start_time1)) # 0.09, there may be small variations for each run
w = np.ascontiguousarray(foot1_hourly.data, dtype=np.float32)
print("footprint to np --- %s seconds ---" % (time.time() - start_time1)) # 3 sec for 2017, 0.1 for 2012

np.sum(foot1_hourly.values != 0)



'''further diagnostics: it is due to the chunksizes when the footprint netcdf was compressed'''
foot1_hourly_2017 = foot1_hourly
# foot1_hourly_2012 = foot1_hourly

enc = foot1_hourly_2017.encoding
print("chunksizes:", enc.get("chunksizes")) #(80, 40, 240)
print("zlib:", enc.get("zlib"), "shuffle:", enc.get("shuffle"))

enc = foot1_hourly_2012.encoding
print("chunksizes:", enc.get("chunksizes")) #(40, 19, 49)
print("zlib:", enc.get("zlib"), "shuffle:", enc.get("shuffle"))



'''revise how I read the netcdf, now only takes 100 sec'''
year = 2017 # 2012 2013 2014 2017
if year in [2012, 2013, 2014]:
    campaign_name = 'carve'
    dir_footprint = f'/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/carve/CARVE_L4_WRF-STILT_Footprint/data/CARVE-{year}-aircraft-footprints-convect/'
else:
    campaign_name = 'arctic_cap'
    dir_footprint = '/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/above/ABoVE_Footprints_WRF_AK_NWCa/data/ArcticCAP_2017_insitu-footprints/'

# read atmospheric observations
df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

# filters for airborne observations
mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
    (df_influence['ABoVE_influence_fraction'] > 0.5) &
    (df_influence['ocean_influence_fraction'] < 0.3) &
    (df_airborne['CO2_change'] < 30) &
    (df_airborne['CO_change'] < 40))[0]

row_id = 29
start_time = time.time()
print(f"Processing row {row_id} of {len(df_airborne)}")

footprint_filename = df_airborne.at[row_id, 'footprint_filename']
with xr.open_dataset(dir_footprint + footprint_filename, cache=False) as ds:
    foot = (
        ds.foot1.rename({'foot1lat':'latitude','foot1lon':'longitude'})
          .transpose('foot1date','latitude','longitude')
          .astype('float32')
          .load()             # <-- actually read & decompress *once*
    )

W = np.ascontiguousarray(foot.data)  # shape (T, lat, lon), fully in RAM

CO2_change = 0.0
month0 = -1; day0 = -1
for i, timestamp in enumerate(foot1date.values):
    print(timestamp)
    start_time1 = time.time()

    timestamp_str = str(timestamp)
    year = int(timestamp_str[:4])
    month = int(timestamp_str[5:7])
    day = int(timestamp_str[8:10])
    hour = int(timestamp_str[11:13])

    # footprint
    w = W[i]
    print("footprint to np --- %s seconds ---" % (time.time() - start_time1))

    '''hourly X-BASE'''
    if day != day0: # if it is on the same day, do not need to read the data again, but this does not save time, the main time consuming part is 
        ds_daily = read_daily_x_base(year, month, day)
        ds_daily = reorganize_xarray(ds_daily)
        ds_daily = mask_ocean_pixels(ds_daily)
        ds_daily = ds_daily.fillna(0)
        print("read daily --- %s seconds ---" % (time.time() - start_time1))
        nee_daily = np.ascontiguousarray(ds_daily['NEE'].data, dtype=np.float32) 
        print("daily to np --- %s seconds ---" % (time.time() - start_time1))

    ds_daily_t = np.tensordot(nee_daily, w, axes=((0,1),(0,1)))
    # ds_daily_t = xr.dot(ds_daily['NEE'], foot1_hourly, dims=["latitude", "longitude"])
    print("daily transported --- %s seconds ---" % (time.time() - start_time1))

    if month != month0: # if it is in the same month, do not need to read the data again
        ds_monthlycycle = read_monthlycycle_x_base(year, month)
        ds_monthlycycle = reorganize_xarray(ds_monthlycycle)
        ds_monthlycycle = mask_ocean_pixels(ds_monthlycycle)
        ds_monthlycycle = ds_monthlycycle.fillna(0)
        # print("read monthly diurnal cycle --- %s seconds ---" % (time.time() - start_time1))

    ds_hourly_deviation = ds_monthlycycle.sel(hour=hour, method="nearest")
    nee_hourly_deviation = np.ascontiguousarray(ds_hourly_deviation['NEE'].data, dtype=np.float32)
    ds_hourly_deviation_t = np.tensordot(nee_hourly_deviation, w, axes=((0,1),(0,1)))

    # ds_hourly_deviation = ds_hourly_deviation.drop_vars("time", errors="ignore")
    # ds_hourly_deviation = ds_hourly_deviation.drop_vars("hour", errors="ignore")
    # ds_hourly_deviation_t = xr.dot(ds_hourly_deviation['NEE'], foot1_hourly, dims=["latitude", "longitude"])
    print("monthly diurnal cycle to np --- %s seconds ---" % (time.time() - start_time1))

    tmp = ds_daily_t + ds_hourly_deviation_t # I decided to separately apply footprint to daily and monthly cycle, then add them together; it takes longer time if I add the two xarray first
    CO2_change += tmp/24/3600/12*1e6 # convert unit from gC m-2 d-1 to μmol m-2 s-1

    # save time info of this timestamp
    month0 = month
    day0 = day

    print("--- %s seconds ---" % (time.time() - start_time1))

print("--- %s seconds ---" % (time.time() - start_time)) 
