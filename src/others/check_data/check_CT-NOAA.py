'''check CT-NOAA at three-hourly and monthly resolutions'''

import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

year = 2012
month = '01'

dir_3hourly = '/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/original/CT-NOAA/three-hourly/'
dir_monthly = '/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/original/CT-NOAA/monthly/'
# dir_3hourly = '/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/CT-NOAA/three-hourly/'


'''3-hourly data'''
files = os.listdir(dir_3hourly)
filtered_files = [f for f in files if f.startswith(f'CT2022.flux1x1.{year}{month}')]
# filtered_files = [f for f in files if f.startswith(f'CT2022.flux0.5x0.5.{year}{month}')]

all_data = []
for file in filtered_files:
    filepath = os.path.join(dir_3hourly, file)
    ds_3hourly = xr.open_dataset(filepath)
    ds_3hourly.time # center of each 3-hourly interval
    ds_daily = ds_3hourly['bio_flux_opt'].mean(dim='time')  # Average over time for each file
    all_data.append(ds_daily)

ds_monthly_aggregated = xr.concat(all_data, dim='file').mean(dim='file')  # Average across all files

# Plot the aggregated monthly data
plt.figure(figsize=(10, 6))
ds_monthly_aggregated.plot()
plt.title('Aggregated Monthly Bio Flux')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


'''monthly data'''
ds_monthly = xr.open_dataset(f'{dir_monthly}CT2022.flux1x1.{year}{month}.nc')['bio_flux_opt']

'''difference'''
ds_diff = ds_monthly - ds_monthly_aggregated

# Plot the difference - exactly the same
plt.figure(figsize=(10, 6))
ds_diff.plot()
plt.title('Aggregated Monthly Bio Flux')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()