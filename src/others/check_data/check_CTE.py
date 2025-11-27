'''check CTE at three-hourly resolution'''

import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

year = 2012
month = '01'

# dir_3hourly = '/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/original/CTE/three-hourly/cte2024_posterior/co2_bio_flux_opt/'; time_var = 'date'
dir_3hourly = '/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/CTE/three-hourly/'; time_var = 'time'

'''3-hourly data'''
files = os.listdir(dir_3hourly)
filtered_files = [f for f in files if f.startswith(f'bio_{year}{month}')]

file = filtered_files[0]
filepath = os.path.join(dir_3hourly, file)
ds_3hourly = xr.open_dataset(filepath)
ds_3hourly[time_var] # start of each 3-hourly interval
ds_3hourly.co2_bio_flux_opt # mol m-2 s-1
ds_daily = ds_3hourly['co2_bio_flux_opt'].mean(dim=time_var)  # Average over time for each file

# Plot the aggregated monthly data
plt.figure(figsize=(10, 6))
ds_daily.plot()
plt.title('Aggregated Monthly Bio Flux')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
