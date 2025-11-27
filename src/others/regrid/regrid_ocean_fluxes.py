import xarray as xr
import numpy as np
import typing
import pandas as pd
import xesmf as xe
import time
from datetime import datetime, timedelta
import os

def regrid_ocean_fluxes(filename) -> None:
    """Regrid GCP-2024 ocean fluxes."""

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

    dataset_name = filename.split('_19')[0]
    print(f"Regridding ocean fluxes '{dataset_name}' data to monthly, half-degree resolution:")

    filename_full = f"/central/groups/carnegie_poc/michalak-lab/data/gcb-2024-ocean/{filename}"
    ds = xr.open_dataset(filename_full)

    # select the period for model benchmarking
    year_start: int = 2000
    year_end: int = 2023
    ds_subset: xr.Dataset = ds[['fgco2']].isel(
        time=(ds.time.dt.year >= year_start)
        & (ds.time.dt.year <= year_end)
    )

    regridder = xe.Regridder(ds_subset, grid_out, "conservative")
    ds_subset_out = regridder(ds_subset)

    # set the time variable
    timestamps: pd.DatetimeIndex = pd.date_range(
    start=f"{year_start}-01-01", end=f"{year_end}-12-31", freq="MS"
    )
    ds_subset_out["time"] = timestamps.to_numpy()

    f_o: str = (
        f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gcb-2024-ocean/global-half-degree/"
        f"{dataset_name}_{year_start}-{year_end}-half-degree.nc"
    )

    ds_subset_out.to_netcdf(
        f_o,
        engine="netcdf4"
    )

    del ds, ds_subset_out, f_o
    print(f"* Ocean fluxes {dataset_name} done")


if __name__ == "__main__":

    dir = '/central/groups/carnegie_poc/michalak-lab/data/gcb-2024-ocean'
    filelist = [f for f in os.listdir(dir) if 'dataprod' in f]
    for filename in filelist:
        regrid_ocean_fluxes(filename)