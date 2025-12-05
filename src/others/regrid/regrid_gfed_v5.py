"""Regrid GFEDv5 fire emissions."""
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import time

def regrid_gfedv5(year: int, regridder) -> None:
    """Regrid GFEDv5 fire emissions."""
    input_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/gfed/v5/"
    compression = dict(zlib=True, complevel=5)

    print(f"Regridding year '{year}' data to monthly, half-degree resolution:")

    if (year <= 2001):
        f = input_dir + f"1997-2001/GFED5_Beta_monthly_{year}.nc"
    else:
        f = input_dir + f"GFED5_Beta_monthly_{year}.nc"
    
    ds = xr.open_dataset(f)
    ds = ds[['CO2']]
    ds_out = regridder(ds)
    print(f"* {year} re-gridded successfully")

    # set the time variable
    timestamps: pd.DatetimeIndex = pd.date_range(
    start=f"{year}-01-01", end=f"{year}-12-31", freq="MS"
    )
    ds_out["time"] = timestamps.to_numpy()


    f_o: str = (
        f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gfed/v5/global-half-degree/"
        f"GFED5_Beta_monthly-half-degree-{year}.nc"
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

    input_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/gfed/v5/"
    f = input_dir + f"GFED5_Beta_monthly_2012.nc"

    ds = xr.open_dataset(f)
    ds = ds[['CO2']]
    # regridder = xe.Regridder(ds, grid_out, "conservative")
    # ds.close()
    # fn = regridder.to_netcdf('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/gfed/regridder_gfedv5.nc')

    # read saved regridder
    fn = xr.open_dataset('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/gfed/regridder_gfedv5.nc')
    regridder = xe.Regridder(ds, grid_out, 'conservative', weights=fn)

    year_start: int = 2002
    year_end: int = 2022

    for year in np.arange(year_start, year_end+1):
        regrid_gfedv5(year=year, regridder=regridder)

    end_time: float = time.time()
    print("Done. This took %.0f seconds" % (end_time - start_time))