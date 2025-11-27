import xarray as xr
import numpy as np
import typing
import pandas as pd
import xesmf as xe
import time
from datetime import datetime, timedelta

def regrid_gridfed(year) -> None:
    """Regrid gridfed fossil fuel emissions."""

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

    print(f"Regridding GridFED {year} data to monthly, half-degree resolution:")

    filename = "/central/groups/carnegie_poc/michalak-lab/data/gcp-gridfed/GCP-GridFEDv2024.0_2012.nc"
    ds_tmp = xr.open_dataset(filename)
    ds0 = xr.open_dataset(filename, group='CO2') # unit: kg CO2 cell-1 month-1

    '''convert unit to kg CO2 m-2 month-1'''
    # calculate cell area in m2
    res_x = 0.1
    res_y = 0.1
    latitudes = ds_tmp['lat'].values
    area = calculate_area(latitudes, res_x, res_y)  # m2
    area_2d = np.tile(area[:, np.newaxis], (1, ds_tmp.sizes['lon']))  # make it 2D
    area_da = xr.DataArray(area_2d, coords=[ds_tmp['lat'], ds_tmp['lon']], dims=['lat', 'lon'])

    ds = ds0[['TOTAL']] / area_da # unit: kg CO2 m-2 month-1

    regridder = xe.Regridder(ds, grid_out, "conservative")
    ds_subset_out = regridder(ds)

    # set the time variable
    timestamps: pd.DatetimeIndex = pd.date_range(
    start=f"{year}-01-01", end=f"{year}-12-31", freq="MS"
    )
    ds_subset_out["time"] = timestamps.to_numpy()
    ds_subset_out['TOTAL'].attrs['units'] = 'kg CO2 m-2 month-1' # add attribute for the unit

    f_o: str = (
        f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gcp-gridfed/global-half-degree/"
        f"GCP-GridFEDv2024.0_{year}-half-degree.nc"
    )

    ds_subset_out.to_netcdf(
        f_o,
        engine="netcdf4"
    )

    del ds, ds_subset_out, f_o
    print(f"* GridFED {year} done")


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


if __name__ == "__main__":
    
    for year in np.arange(2012,2024):
        regrid_gridfed(year)