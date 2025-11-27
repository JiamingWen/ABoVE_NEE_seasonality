'''regrid X-BASE monthlycycle and daily output from 0.25 degree to 0.5 degree'''
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import time
import pyproj
from pyproj import CRS

def regrid_x_base(varname: str, res_name: str, year: int, regridder) -> None:
    """Regrid X-BASE"""
    input_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/fluxcom-x-base/"
    compression = dict(zlib=True, complevel=5)

    print(f"Regridding year '{year}' data to half-degree resolution:")

    f = f"{input_dir}{res_name}/{varname}_{year}_025_{res_name}.nc"

    ds0 = xr.open_dataset(f, decode_coords="all")
    ds = ds0[[varname]]

    ds = ds.rename({"lon": "longitude", "lat": "latitude"})

    # Update longitude attributes - somehow the original file has wrong attributes
    ds['longitude'].attrs['long_name'] = 'longitude'
    ds['longitude'].attrs['standard_name'] = 'longitude'
    
    ds_out = regridder(ds)
    print(f"* {year} {res_name} re-gridded successfully")

    f_o: str = (
        f"{input_dir}{res_name}/{varname}_{year}_05_{res_name}.nc"
    )
    ds_out.to_netcdf(
        f_o,
        engine="netcdf4",
        encoding={v: compression for v in ds_out.data_vars},
    )

    del ds0, ds, ds_out, f_o
    print(f"* {year} done")

if __name__ == "__main__":

    # generate the regridder once and re-use it

    # grid for output data
    res_out: float = 0.5
    grid_out: xr.Dataset = xr.Dataset(
        {
            "latitude": (
                ["latitude"],
                np.linspace(
                    -90.0 + 0.5 * res_out,
                    90.0 - 0.5 * res_out,
                    round(180.0 / res_out),
                ),
            ),
            "longitude": (
                ["longitude"],
                np.linspace(
                    -180.0 + 0.5 * res_out,
                    180.0 - 0.5 * res_out,
                    round(360.0 / res_out),
                ),
            ),
        }
    )

    input_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/fluxcom-x-base/"
    f = f"{input_dir}monthlycycle/NEE_2012_025_monthlycycle.nc"

    ds0 = xr.open_dataset(f, decode_coords="all")
    ds = ds0[['NEE']]

    ds = ds.rename({"lon": "longitude", "lat": "latitude"})

    # Update longitude attributes - somehow the original file has wrong attributes
    ds['longitude'].attrs['long_name'] = 'Longitude'
    ds['longitude'].attrs['standard_name'] = 'longitude'
    ds['longitude'].attrs['units'] = 'degrees_east'

    # # save regridder
    regridder = xe.Regridder(ds, grid_out, "conservative")
    fn = regridder.to_netcdf('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/x_base_hourly/regridder_025_to_05.nc')

    # read saved regridder
    fn = xr.open_dataset('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/x_base_hourly/regridder_025_to_05.nc')
    regridder = xe.Regridder(ds, grid_out, 'conservative', weights=fn)

    # loop through the variables and years
    start_time: float = time.time()
    # year_start: int = 2001
    # year_end: int = 2020
    for varname in ['NEE']:
        for res_name in ['monthlycycle', 'daily']:
            for year in [2012, 2013, 2014, 2017]:
                regrid_x_base(varname=varname, res_name=res_name, year=year, regridder=regridder)

    end_time: float = time.time()
    print("Done. This took %.0f seconds" % (end_time - start_time))