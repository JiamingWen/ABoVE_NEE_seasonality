"""Regridding fPAR and LAI to half-degree"""
import xarray as xr
import numpy as np
import typing
import pandas as pd
import xesmf as xe
import time
from datetime import datetime, timedelta


def regrid_fpar_lai(varname, year) -> None:
    """Regrid MODIS fpar or lai."""
    compression = dict(zlib=True, complevel=5)

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

    print(f"Regridding MODIS {varname} data to monthly, half-degree resolution:")

    # read the data
    filename1 = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/modis_fpar/{varname}-500m-monthly-{year}04-{year}11.nc'
    ds = xr.open_dataset(filename1)

    # change coordinate names
    ds = ds.rename(
        {"month": "time", "y": "latitude", "x": "longitude"}
    )
    
    # select the period for model benchmarking
    ds_subset = ds

    # change nan to zero - otherwise the value at coarse grids will be zero if any nan value falls into it
    ds_subset = ds_subset.fillna(0)


    regridder = xe.Regridder(ds_subset, grid_out, "conservative")
    ds_subset_out = regridder(ds_subset)

    # set the time variable
    timestamps: pd.DatetimeIndex = pd.date_range(
    start=f"{year}-04-01", end=f"{year}-11-30", freq="MS"
    )
    ds_subset_out["time"] = timestamps.to_numpy()

    f_o: str = (
        f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/modis_fpar/{varname}-half-degree-monthly-{year}04-{year}11.nc"
    )
    ds_subset_out.to_netcdf(
        f_o,
        engine="netcdf4",
        encoding={v: compression for v in ds_subset_out.data_vars},
    )

    del ds, ds_subset_out, f_o
    print(f"* {varname} done")

if __name__ == "__main__":
    start_time: float = time.time()
    for varname in ['Fpar', 'Lai']:
        for year in [2012, 2013, 2014]: #, 2017
            print(year, varname)
            regrid_fpar_lai(varname=varname, year=year)

    end_time: float = time.time()
    print("Done. This took %.0f seconds" % (end_time - start_time))