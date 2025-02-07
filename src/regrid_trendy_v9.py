"""Regrid carbon flux estimates."""
import glob
import gzip
import os
import sys
import time
import typing
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

import config.io
from config import dirs
from libs.misc import format_running_time

if os.path.isdir(dirs.libflux):
    sys.path.append(dirs.libflux)
    from libflux.data_io import read_mat
    from libflux.const import molar_mass_c, seconds_per_day
else:
    raise ImportError("Dependency `libflux` not found!")


def regrid_trendy(var_name: str = "nbp") -> None:
    """Regrid TRENDY v9 carbon flux estimates."""
    input_dir: str = f"{dirs.trendy_v9}/output/S3"
    compression = config.io.netcdf_compression

    year_start: int = 2012
    year_end: int = 2019

    flist: List[str] = sorted(
        glob.glob(f"{input_dir}/*_{var_name}.nc")
        + glob.glob(f"{input_dir}/*_{var_name}.nc.gz")
    )

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

    print(f"Regridding '{var_name}' data to monthly, half-degree resolution:")
    for f in flist:
        model_name: str = os.path.basename(f).split(".nc")[0]
        if (
            model_name.startswith("CABLE-POP")
            or model_name.startswith("DLEM")
            or model_name.startswith("LPJ-GUESS")
        ):
            # notes
            # 1. CABLE-POP results have been withdrawn from TRENDY v9
            # 2. DLEM does not have monthly output
            # 3. LPJ-GUESS does not have monthly output of net biome
            #    productivity
            continue

        if f.endswith(".gz") and (not model_name.startswith("VISIT")):
            # use gzip to read data except for the VISIT model
            gz = gzip.open(f)
            ds: xr.Dataset = xr.open_dataset(gz)  # type: ignore
        else:
            try:
                ds = xr.open_dataset(f)
            except ValueError:
                ds = xr.open_dataset(f, decode_times=False)
                # detect the name of time coordinate
                time_name: str = [
                    typing.cast(str, coord)
                    for coord in ds.coords
                    if typing.cast(str, coord).startswith("time")
                ][0]
                if time_name != "time":
                    ds = ds.rename({time_name: "time"})

                _time_start_str: str = ds.time.units.split("since")[1]
                _time_start: pd.Timestamp = (
                    pd.Timestamp(_time_start_str)
                    + pd.offsets.MonthEnd(0)
                    + pd.offsets.MonthBegin(-1)
                )
                _timestamps: pd.DatetimeIndex = pd.date_range(
                    start=_time_start, periods=len(ds.time), freq="MS"
                )
                ds = ds.assign_coords(time=_timestamps)
            except AttributeError:
                print(f"* {model_name} failed")
                continue

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

        # detect the original resolution
        res_lat: int = ds.latitude.values[1] - ds.latitude.values[0]
        res_lon: int = ds.longitude.values[1] - ds.longitude.values[0]

        # select the period for model benchmarking
        ds_subset: xr.Dataset = ds.isel(
            time=(ds.time.dt.year >= year_start)
            & (ds.time.dt.year <= year_end),
            latitude=(ds.latitude >= -90) & (ds.latitude <= 90),
        )

        if "ISBA-CTRIP" in model_name:
            ds_subset = ds_subset.drop(["lon_FULL_bnds", "lat_FULL_bnds"])
            # fill out Antarctica with NaN values for this model
            lats_addenda: np.ndarray = np.linspace(-89.5, -60.5, 30)
            data_addenda: np.ndarray = np.full(
                (
                    len(ds_subset["time"]),
                    len(lats_addenda),
                    len(ds_subset["longitude"]),
                ),
                np.nan,
            )
            ds_subset_addenda: xr.Dataset = xr.Dataset(
                data_vars={
                    var_name: (["time", "latitude", "longitude"], data_addenda)
                },
                coords={
                    "time": ds_subset["time"],
                    "latitude": (["latitude"], lats_addenda),
                    "longitude": ds_subset["longitude"],
                },
                attrs=ds_subset.attrs,
            )
            ds_subset_addenda["latitude"].attrs = ds_subset["latitude"].attrs
            ds_subset = xr.concat(
                [ds_subset_addenda, ds_subset], dim="latitude"
            )
            ds_subset["latitude"].attrs.pop("bounds")
            ds_subset["longitude"].attrs.pop("bounds")

        if np.isclose(res_lat, res_out) and np.isclose(res_lon, res_out):
            # no need to regrid if the grid is already one-degree
            ds_subset_out: xr.Dataset = ds_subset
        else:
            # regrid to half-degree resolution
            regridder = xe.Regridder(ds_subset, grid_out, "conservative")
            ds_subset_out = regridder(ds_subset)

        # harmonize the timestamps: keep the days on the first of each month
        # note: this will alter non-standard calendars, for example, a calendar
        # without a leap day
        timestamps: pd.DatetimeIndex = pd.date_range(
            start=f"{year_start}-01-01", end=f"{year_end}-12-31", freq="MS"
        )
        ds_subset_out["time"] = timestamps.to_numpy()

        f_o: str = (
            f"{dirs.data}/input/trendy-v9/global-half-degree/"
            f"{model_name}-half-degree.nc"
        )
        ds_subset_out.to_netcdf(
            f_o,
            engine="netcdf4",
            encoding={v: compression for v in ds_subset_out.data_vars},
        )

        # must close the gz file
        if f.endswith(".gz"):
            gz.close()

        del ds, ds_subset, ds_subset_out, f_o
        print(f"* {model_name} done")


def regrid_trendy_v6() -> None:
    """Regrid TRENDY v6 carbon flux estimates."""
    input_dir: str = (
        f"{dirs.root}/../nasa-ids/data/exp-vars/raw/ids-new/TRENDYv6_s2007"
    )
    output_dir: str = f"{dirs.trendy_v6}"
    compression = config.io.netcdf_compression
    start_date: str = "2007-01"

    # lower and upper limits for data value filtering
    lwr_lim: float = -1e8
    upr_lim: float = 1e8

    # define the selected scenario
    scenario: str = "S3"  # co2, climate, and land use change

    # define the coordinates - half-degree global grid
    lons: np.ndarray = np.arange(-179.75, 180.0, 0.5)
    lats: np.ndarray = np.arange(-89.75, 90.0, 0.5)

    mat_list: List[str] = sorted(glob.glob(f"{input_dir}/*.mat"))
    da_list: List[xr.DataArray] = []
    for f in mat_list:
        data_dict: Optional[Dict[str, np.ndarray]] = read_mat(f)
        if data_dict:
            data: np.ndarray = data_dict[next(iter(data_dict))]
            data[(data < lwr_lim) | (data > upr_lim)] = np.nan
            # parse model and variable names
            f_split: List[str] = os.path.splitext(os.path.basename(f))[
                0
            ].split("_")
            f_split_no_year: List[str] = [
                s if s != "gpp" else s.upper()
                for s in f_split
                if not s.isdigit()
            ]
            if len(f_split_no_year) == 1:
                model_var: str = f_split_no_year[0]
            else:
                model_var = "::".join(
                    [f_split_no_year[0], f_split_no_year[-1]]
                )

            # flip the sign of NBP or NEP to get NEE
            if "::nbp" in model_var:
                data = -data
                model_var = model_var.replace("::nbp", "::NEE")
            elif "::nep" in model_var:
                data = -data
                model_var = model_var.replace("::nep", "::NEE")

            timestamps: pd.DatetimeIndex = pd.date_range(
                start=start_date, periods=data.shape[2], freq="MS"
            )

            # note: latitudes were arranged from north to south
            # need to flip the latitude when packaging the data array
            da: xr.DataArray = xr.DataArray(
                np.moveaxis(data[::-1, :, :], -1, 0),
                name=model_var,
                coords={
                    "time": (["time"], timestamps),
                    "lat": (["lat"], lats),
                    "lon": (["lon"], lons),
                },
                dims=["time", "lat", "lon"],
            )
            da_list.append(da)

    ds: xr.Dataset = xr.merge(da_list)

    # convert flux unit from [kgC m^-2 month^-1] to [micromol m^-2 s^-1]
    days_in_month: np.ndarray = ds.time.dt.days_in_month.values
    conv_fac: np.ndarray = 1e6 / days_in_month * molar_mass_c * seconds_per_day
    ds = ds * conv_fac.reshape((-1, 1, 1))

    # add attributes for each data variable
    #
    # NOTE: attrs can only be added after the dataset is merged because
    # they are not preserved. This seems to be related to issue
    # [#2245](https://github.com/pydata/xarray/issues/2245) of xarray.
    for _v in ds.data_vars:
        # add dataset group name, because this allows grouping data
        # variables with `filter_by_attrs`
        v = typing.cast(str, _v)
        ds[v].attrs.update({"ensemble": "trendy-v6"})
        # add variable types as attributes
        # for prognostic models, add scenario names as attributes
        model_name: str = v.split("::")[0]
        var_type: str = v.split("::")[1]
        unit: str = "Âµmol m^-2 s^-1"
        ds[v].attrs.update(
            {
                "model": model_name,
                "scenario": scenario,
                "var_type": var_type,
                "unit": unit,
            }
        )

    # name the output file
    year_start, year_end = ds.time.isel(time=[0, -1]).dt.year.values
    f_o: str = (
        f"{output_dir}/trendy-v6-{year_start}-to-{year_end}"
        "-global-halfdeg-monthly.nc"
    )
    # save to netcdf
    ds.to_netcdf(
        f_o,
        engine="netcdf4",
        encoding={v: compression for v in ds.data_vars},
    )
    print(f"Regridded TRENDY v6 data sets saved to {f_o}.")


if __name__ == "__main__":
    start_time: float = time.time()
    regrid_trendy_v6()  # legacy data from NASA IDS 2016 project
    # TRENDY v9
    for var_name in ["gpp", "nbp", "ra", "rh"]:
        regrid_trendy(var_name=var_name)

    end_time: float = time.time()
    print("Done. " + format_running_time(end_time - start_time))
