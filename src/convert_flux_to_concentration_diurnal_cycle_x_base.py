'''
convert the X-BASE monthly diurnal cycle output to the concentration space
later it may be imposed to different datasets
'''

import math
import os
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
import numpy as np
import pandas as pd
import xarray as xr
from multiprocessing import Pool, cpu_count, get_context
from functools import partial
import time
from functools import lru_cache
import os
os.chdir('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import read_x_base_monthlycycle

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
    '/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/above_mask/ocean-mask-half-degree.nc'
)
_OCEAN_MASK = _OCEAN_MASK.assign_coords(
    longitude=((( _OCEAN_MASK.longitude + 180) % 360) - 180)
).sortby('longitude')

def mask_ocean_pixels (ds):
    # only select land pixels - this may lead to small differences in CO2 enhancement (e.g., 0.08 ppm)
    return ds.where(_OCEAN_MASK['seamask'] == 0)

year = 2012 # 2012 2013 2014 2017
if year in [2012, 2013, 2014]:
    campaign_name = 'carve'
    dir_footprint = f'/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/carve/CARVE_L4_WRF-STILT_Footprint/data/CARVE-{year}-aircraft-footprints-convect/'
else:
    campaign_name = 'arctic_cap'
    dir_footprint = '/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/above/ABoVE_Footprints_WRF_AK_NWCa/data/ArcticCAP_2017_insitu-footprints/'

# read atmospheric observations
df_airborne = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
df_influence = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

# filters for airborne observations
mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
    (df_influence['ABoVE_influence_fraction'] > 0.5) &
    (df_influence['ocean_influence_fraction'] < 0.3) &
    (df_airborne['CO2_change'] < 30) &
    (df_airborne['CO_change'] < 40))[0]

# Output paths
out_csv = f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_X-BASE-monthly_diurnal.csv'
tmp_csv = out_csv + ".tmp"
result_df = pd.DataFrame({'X-BASE': np.nan}, index=range(len(df_airborne)))


'''in parallel (fast)'''
def process_row(row_id, footprint_filename, dir_footprint, year):
    """Process a single row for CO2 change calculation."""
    # print(f"Processing row {row_id}")
    with xr.open_dataset(dir_footprint + footprint_filename, cache=False) as ds:
        foot1 = (
            ds.foot1.rename({'foot1lat':'latitude','foot1lon':'longitude'})
            .transpose('foot1date','latitude','longitude')
            .astype('float32')
            .load()             # <-- actually read & decompress *once*
        )
        foot1date = ds.foot1date

    W = np.ascontiguousarray(foot1.data)  # shape (T, lat, lon), fully in RAM

    CO2_change = 0.0
    month0 = -1
    for i, timestamp in enumerate(foot1date.values):

        timestamp_str = str(timestamp)
        year = int(timestamp_str[:4])
        month = int(timestamp_str[5:7])
        day = int(timestamp_str[8:10])
        hour = int(timestamp_str[11:13])

        # footprint
        w = W[i]

        '''X-BASE monthly diurnal cycle'''
        
        if month != month0: # if it is in the same month, do not need to read the data again
            ds_monthlycycle = read_x_base_monthlycycle(year, month)
            ds_monthlycycle = reorganize_xarray(ds_monthlycycle)
            ds_monthlycycle = mask_ocean_pixels(ds_monthlycycle)
            ds_monthlycycle = ds_monthlycycle.fillna(0)

        ds_hourly = ds_monthlycycle.sel(hour=hour, method="nearest")
        nee_hourly = np.ascontiguousarray(ds_hourly['NEE'].data, dtype=np.float32)
        ds_hourly_t = np.tensordot(nee_hourly, w, axes=((0,1),(0,1)))

        CO2_change += ds_hourly_t/24/3600/12*1e6 # convert unit from gC m-2 d-1 to μmol m-2 s-1

        # save time info of this timestamp
        month0 = month
        day0 = day

    return row_id, CO2_change


# Build task list (default: all)
tasks = list(mask_id)

# If resuming, overwrite 'tasks' to skip completed rows
if os.path.exists(out_csv):
    # Resume: load existing results
    prev = pd.read_csv(out_csv)
    # align shape if needed
    if len(prev) == len(result_df) and 'X-BASE' in prev.columns:
        result_df['X-BASE'] = prev['X-BASE'].values
        print(f"[resume] Loaded {out_csv}. "
              f"Completed rows: {np.isfinite(result_df['X-BASE']).sum()}/{len(result_df)}")
    else:
        print("[resume] Existing file shape mismatch; starting a fresh result_df.")

    # Build remaining task list (skip completed)
    remaining_mask = np.isnan(result_df['X-BASE'].values)
    tasks = [rid for rid in mask_id if remaining_mask[rid]]
    print(f"[run] Will process {len(tasks)} rows (skipping {len(mask_id) - len(tasks)} already done).")

def _worker(args):
    # args is (row_id, footprint_filename, dir_footprint, year)
    return process_row(*args)

def main():
    start_time = time.time()
    footprint_filenames = df_airborne['footprint_filename'].tolist()

    # pool settings
    procs = 10
    chunksize = 8
    # chunksize = 1 if len(tasks) <= 64 else max(1, math.ceil(len(tasks)/(4*procs)))

    args_iter = ((rid, footprint_filenames[rid], dir_footprint, year) for rid in tasks)

    # Checkpoint cadence
    CHECK_EVERY = 50  # write every 50 results; tune as you like
    FLUSH_SEC   = 300   # 5 minutes
    last_flush  = time.time()
    ckpt = {"written": 0, "processed_since_flush": 0}

    def flush_checkpoint():
        # atomic write: tmp then replace
        result_df.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, out_csv)
        ckpt["written"] += ckpt["processed_since_flush"]
        print(f"[ckpt] Wrote checkpoint "
            f"({ckpt['written']} new rows so far) at {time.strftime('%H:%M:%S')}")

    with get_context("spawn").Pool(processes=procs, maxtasksperchild=100) as pool:
        printed = 0
        for row_id, val in pool.imap_unordered(_worker, args_iter, chunksize=chunksize):
            result_df.at[row_id, 'X-BASE'] = float(val)

            if printed < 3:
                print(f"[first] completed row_id={row_id}", flush=True)
                printed += 1

            ckpt["processed_since_flush"] += 1
            now = time.time()
            if ckpt["processed_since_flush"] >= CHECK_EVERY  or (now - last_flush) > FLUSH_SEC:
                flush_checkpoint()
                ckpt["processed_since_flush"] = 0
                last_flush = now

    if ckpt["processed_since_flush"]:
        flush_checkpoint()

    result_df.to_csv(out_csv, index=False)
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))

if __name__ == "__main__":
    main()

'''in parallel (fast) ends'''


# '''check output'''
# df_diurnal_cycle = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_magnitude/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_X-BASE-monthly_diurnal.csv')
# df_daily = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_magnitude/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_X-BASE-daily.csv')
# df_hourly = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_magnitude/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_X-BASE-hourly.csv')
# print(len(mask_id), np.sum(df_diurnal_cycle['X-BASE_CO2_change'].notna()), np.sum(df_daily['X-BASE_CO2_change'].notna()), np.sum(df_hourly['X-BASE_CO2_change'].notna()))
# diff = abs(df_diurnal_cycle['X-BASE_CO2_change'] + df_daily['X-BASE_CO2_change'] - df_hourly['X-BASE_CO2_change'])
# diff[~pd.isna(diff)]

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# date_time = pd.to_datetime(df_airborne['footprint_time_UTC'], errors='coerce')
# plt.plot(date_time, diff, marker='o', linestyle='-', color='b')
# plt.show()

# date_time[diff>0.1]

# max_diff_index = diff.idxmax()
# print(f"Largest difference is at index {max_diff_index} with a value of {diff[max_diff_index]}")
# '''check output ends'''
