"""Regrid upscaled ABCflux"""

import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import time
import rioxarray
import pyproj
from pyproj import CRS

def regrid_abcflux(varname: str, year: int, regridder) -> None:
    """Regrid upscaled ABCflux"""
    input_dir: str = "/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/abcflux_upscaled/CO2Fluxes_Arctic_Boreal_Domain/data/"
    compression = dict(zlib=True, complevel=5)

    print(f"Regridding year '{year}' data to monthly, half-degree resolution:")

    f = input_dir + f"CO2Fluxes_Arctic_Boreal_{varname}_{year}.nc"

    ds0 = xr.open_dataset(f, decode_coords="all")
    ds_variable = ds0[[varname]]
    ds = ds_variable.rio.reproject("EPSG:4326")

    ds = ds.rename({'tm': "time"})
    timestamps: pd.DatetimeIndex = pd.date_range(
        start=f"{year}-01-01", end=f"{year}-12-31", freq="MS"
    )
    ds["time"] = timestamps.to_numpy()

    ds = ds.rename({"x": "longitude", "y": "latitude"})

    # change nan to zero - otherwise the value at coarse grids will be zero if any nan value falls into it
    ds = ds.fillna(0)

    ds_out = regridder(ds)
    print(f"* {year} re-gridded successfully")

    # set the time variable
    timestamps: pd.DatetimeIndex = pd.date_range(
    start=f"{year}-01-01", end=f"{year}-12-31", freq="MS"
    )
    ds_out["time"] = timestamps.to_numpy()

    f_o: str = (
        f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/abcflux_upscaled/half-degree/"
        f"CO2Fluxes_Arctic_Boreal_{varname}_{year}-half-degree.nc"
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

    input_dir: str = "/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/abcflux_upscaled/CO2Fluxes_Arctic_Boreal_Domain/data/"
    f = input_dir + f"CO2Fluxes_Arctic_Boreal_NEE_2001.nc"

    ds0 = xr.open_dataset(f, decode_coords="all")
    ds_variable = ds0[['NEE']]
    ds = ds_variable.rio.reproject("EPSG:4326")

    ds = ds.rename({'tm': "time"})
    timestamps: pd.DatetimeIndex = pd.date_range(
        start=f"2001-01-01", end=f"2001-12-31", freq="MS"
    )
    ds["time"] = timestamps.to_numpy()

    ds = ds.rename({"x": "longitude", "y": "latitude"})

    # # save regridder
    # regridder = xe.Regridder(ds, grid_out, "conservative")
    # fn = regridder.to_netcdf('regridder.nc')

    # read saved regridder
    fn = xr.open_dataset('regridder.nc')
    regridder = xe.Regridder(ds, grid_out, 'conservative', weights=fn)

    # loop through the variables and years
    start_time: float = time.time()
    year_start: int = 2013
    year_end: int = 2020
    for varname in ['Reco']: #'NEE', 'GPP', 
        for year in np.arange(year_start, year_end+1):
        # for year in [2012, 2013, 2014, 2017]:
            regrid_abcflux(varname=varname, year=year, regridder=regridder)

    end_time: float = time.time()
    print("Done. This took %.0f seconds" % (end_time - start_time))