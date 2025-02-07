"""Regrid ODIAC land fossil fuel emissions."""
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import time

def regrid_odiac(year: int) -> None:
    """Regrid ODIAC land fossil fuel emissions."""
    input_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/odiac2022_1x1d_2000to2021/"
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

    print(f"Regridding year '{year}' data to monthly, half-degree resolution:")

    f = input_dir + f"odiac2022_1x1d_{year}.nc"
    ds = xr.open_dataset(f)
    ds = ds.rename({'month': "time"})
    timestamps: pd.DatetimeIndex = pd.date_range(
        start=f"{year}-01-01", end=f"{year}-12-31", freq="MS"
    )
    ds["time"] = timestamps.to_numpy()

    regridder = xe.Regridder(ds, grid_out, "conservative")
    ds_out = regridder(ds)
    print(f"* {year} re-gridded successfully")

    # set the time variable
    timestamps: pd.DatetimeIndex = pd.date_range(
    start=f"{year}-01-01", end=f"{year}-12-31", freq="MS"
    )
    ds_out["time"] = timestamps.to_numpy()

    f_o: str = (
        f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/odiac/global-half-degree/"
        f"odiac2022-half-degree-{year}.nc"
    )
    ds_out.to_netcdf(
        f_o,
        engine="netcdf4",
        encoding={v: compression for v in ds_out.data_vars},
    )

    del ds, ds_out, f_o
    print(f"* {year} done")

if __name__ == "__main__":
    start_time: float = time.time()
    year_start: int = 2000
    year_end: int = 2021
    for year in np.arange(year_start, year_end+1):
        regrid_odiac(year=year)

    end_time: float = time.time()
    print("Done. This took %.0f seconds" % (end_time - start_time))