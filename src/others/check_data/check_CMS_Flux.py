'''check 3-hourly outputs of CMS-Flux'''
import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

filepath = '/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/original/CMS-Flux/three-hourly'
filename = 'CMS-Flux.GCP.3hrly.grid.1x1.2010-2022.nc'
ds = xr.open_dataset(os.path.join(filepath, filename))

varable = ds['3-hourly-prior']
ds.ntime1

# Select a specific pixel (e.g., latitude=0, longitude=0)
lat0 = 42  # Replace with the desired latitude index
lon0 = -76  # Replace with the desired longitude index

# Extract data for January 2010 (2010.01)
pixel_data = varable.sel(ntime1=slice('2010-01-01', '2010-01-31')).isel(lat=lat0, lon=lon0)

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(pixel_data['ntime1'], pixel_data, marker='o', linestyle='-')
plt.xlabel('Time')
plt.ylabel('Variable Value')
plt.grid()
plt.show()

np.mean(pixel_data)