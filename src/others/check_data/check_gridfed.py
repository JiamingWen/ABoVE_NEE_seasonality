'''read GridFED data'''

import xarray as xr

filename = "/central/groups/carnegie_poc/michalak-lab/data/gcp-gridfed/GCP-GridFEDv2024.0_2012.nc"
ds = xr.open_dataset(filename)
print(ds)
ds.lat # 0.1 degree
ds.time # only for one year
print(ds.data_vars) # empty


import netCDF4 as nc
import numpy as np
f = nc.Dataset(filename)
print(f.groups.keys())  # the data is in the groups


ds2 = xr.open_dataset(filename, group='CO2')
print(ds2)
ds2.lat # 0.1 degree
ds2.time # only for one year
print(ds2.data_vars)
ds2['TOTAL'] #unit: kg CO2 cell-1 month-1

# calculate annual sum
total_sum = ds2['TOTAL'].sum().item()
print(f"Sum of TOTAL: {total_sum/1e12} Pg CO2")  # convert to Pg CO2



# calculate cell area in m2
def calculate_area(latitudes, res_x, res_y):
    """Calculate grid cell area in m2."""
    re = 6371220  # Earth radius in meters
    rad = np.pi / 180.0  # Radians per degree
    con = re * rad  # Distance per degree
    clat = np.cos(latitudes * rad)  # Cosine of latitude in radians
    dlon = res_x  # Longitude resolution
    dlat = res_y  # Latitude resolution
    dx = con * dlon * clat  # dx at each latitude
    dy = con * dlat  # dy is constant
    dxdy = dy * dx  # Area of each grid cell
    return dxdy

res_x = 0.1
res_y = 0.1
latitudes = ds['lat'].values
area = calculate_area(latitudes, res_x, res_y)  # m2
area_2d = np.tile(area[:, np.newaxis], (1, ds.dims['lon']))  # make it 2D
area_da = xr.DataArray(area_2d, coords=[ds['lat'], ds['lon']], dims=['lat', 'lon'])

ds2[['TOTAL']] / area_da # unit: kg CO2 m-2 month-1