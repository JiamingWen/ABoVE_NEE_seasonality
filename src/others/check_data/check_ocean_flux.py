'''check ocean fluxes provided in GCB2024'''

import os
import xarray as xr

dir = '/central/groups/carnegie_poc/michalak-lab/data/gcb-2024-ocean'
filelist = [f for f in os.listdir(dir) if 'dataprod' in f]
filename = f'{dir}/{filelist[0]}'

ds = xr.open_dataset(filename)
print(ds)
for var in ds.data_vars:
    print(f"Variable: {var}")
    print(ds[var].attrs)