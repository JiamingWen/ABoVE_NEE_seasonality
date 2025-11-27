"""Regrid CERES PAR"""
import xarray as xr
import numpy as np
import typing
import pandas as pd
import xesmf as xe
import time
from datetime import datetime, timedelta


def regrid_ceres_par() -> None:
    """Regrid CERES PAR."""
    compression = dict(zlib=True, complevel=5)

    year_start: int = 2001
    year_end: int = 2020

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

    print(f"Regridding CERES PAR data to monthly, half-degree resolution:")

    # read the data
    filename1 = '/central/groups/carnegie_poc/michalak-lab/data/ceres-par/CERES_SYN1deg-Month_Terra-Aqua-MODIS_Ed4.1_Subset_200003-202312.nc'
    ds = xr.open_dataset(filename1)

    # harmonize coordinate names
    time_name = [
        typing.cast(str, coord)
        for coord in ds.coords
        if typing.cast(str, coord).startswith("time")
    ][0]
    lat_name: str = [
        typing.cast(str, coord)
        for coord in ds.coords
        if typing.cast(str, coord).startswith("lat")
    ][0]
    lon_name: str = [
        typing.cast(str, coord)
        for coord in ds.coords
        if typing.cast(str, coord).startswith("lon")
    ][0]
    ds = ds.rename(
        {time_name: "time", lat_name: "latitude", lon_name: "longitude"}
    )
    
    # select the period for model benchmarking
    ds["PAR"] = ds["adj_sfc_par_direct_all_mon"] + ds["adj_sfc_par_diff_all_mon"]
    var_list = ['PAR', 'adj_sfc_par_direct_all_mon', 'adj_sfc_par_diff_all_mon']
    ds_subset: xr.Dataset = ds[var_list].isel(
        time=(ds.time.dt.year >= year_start)
        & (ds.time.dt.year <= year_end),
        latitude=(ds.latitude >= -90) & (ds.latitude <= 90),
    )

    regridder = xe.Regridder(ds_subset, grid_out, "conservative")
    ds_subset_out = regridder(ds_subset)

    # set the time variable
    timestamps: pd.DatetimeIndex = pd.date_range(
    start=f"{year_start}-01-01", end=f"{year_end}-12-31", freq="MS"
    )
    ds_subset_out["time"] = timestamps.to_numpy()

    f_o: str = (
        f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/ceres-par/CERES_PAR-half-degree.nc"
    )
    ds_subset_out.to_netcdf(
        f_o,
        engine="netcdf4",
        encoding={v: compression for v in ds_subset_out.data_vars},
    )

    del ds, ds_subset_out, f_o
    print(f"* CERES PAR done")

if __name__ == "__main__":
    start_time: float = time.time()
    regrid_ceres_par()

    end_time: float = time.time()
    print("Done. This took %.0f seconds" % (end_time - start_time))