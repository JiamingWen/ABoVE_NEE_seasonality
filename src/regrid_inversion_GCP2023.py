import xarray as xr
import numpy as np
import typing
import pandas as pd
import xesmf as xe
import time
from datetime import datetime, timedelta

def regrid_inversions_GCP2023(dataset_name, ds, is_posterior) -> None:
    """Regrid inversions."""

    year_start: int = 2000
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

    print(f"Regridding '{dataset_name}' data to monthly, half-degree resolution:")
    
    # select the period for model benchmarking
    ds_subset: xr.Dataset = ds.isel(
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

    # set variable name
    ds_subset_out.name = ds.name

    # copy the attributes
    ds_subset_out = ds_subset_out.assign_attrs(ds.attrs)

    if is_posterior == False:
        posterior_str = '-prior'
    elif is_posterior == True:
        posterior_str = ''

    f_o: str = (
        f"/central/groups/carnegie_poc/michalak-lab/data/inversions/inversions-half-degree/GCP2023{posterior_str}/"
        f"{dataset_name}-half-degree.nc"
    )

    ds_subset_out.to_netcdf(
        f_o,
        engine="netcdf4"
    )

    del ds, ds_subset_out, f_o
    print(f"* {dataset_name} done")

ds = xr.open_dataset('/central/groups/carnegie_poc/michalak-lab/data/inversions/inversions_raw/GCP2023_inversions_1x1_version1_1_20240124.nc')
for inversion_num in np.arange(ds.ensemble_member_name.shape[0]):
    
    inversion_name = ''.join(ds.ensemble_member_name[inversion_num].values)
    
    for is_posterior in [False, True]:
        if is_posterior == False: # prior
            land_flux_indvidual = ds.prior_flux_land[inversion_num]
        elif is_posterior == True: # posterior
            land_flux_indvidual = ds.land_flux_only_fossil_cement_adjusted[inversion_num] #unit: PgC/m2/yr

        regrid_inversions_GCP2023(inversion_name, land_flux_indvidual, is_posterior)