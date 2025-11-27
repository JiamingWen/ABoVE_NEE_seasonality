# read GFEDv5 fire emission data

import xarray as xr

year = 2012
month = 1

input_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/gfed/v5/"
if (year <= 2001): # 1 degree
    filename = input_dir + f"1997-2001/GFED5_Beta_monthly_{year}.nc"
else: # 0.25 degree
    filename = input_dir + f"GFED5_Beta_monthly_{year}.nc"

# Open the NetCDF file
ds = xr.open_dataset(filename)
print(ds)
ds.time
ds.lat
ds.CO2 #unit: g CO2 m^-2 month^-1