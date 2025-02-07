"""Regrid TRENDY v11 carbon flux estimates."""
import glob
import os
import typing
from typing import List
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import time

def regrid_trendy(var_name: str = "nbp") -> None:
    """Regrid TRENDY v11 output"""
    input_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/trendy-v11/output/S3"
    # compression = config.io.netcdf_compression
    compression = dict(zlib=True, complevel=5)

    year_start: int = 2012
    year_end: int = 2021

    flist: List[str] = sorted(
        glob.glob(f"{input_dir}/*_{var_name}.nc")
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

    # remove ISAM from tsl flist for now
    if var_name == 'tsl':
        flist = [filename for filename in flist if 'ISAM' not in filename]

    # only keep four selected models for carbon pools
    if var_name in ['cLitter', 'cSoil']:
        selected_models = ['CLASSIC', 'ISBA-CTRIP', 'JSBACH', 'JULES']
        flist = [filename for filename in flist if any(selected_model in filename for selected_model in selected_models)]

    for f in flist:
        model_name: str = os.path.basename(f).split(".nc")[0]
        if '_Monthly' in model_name:
            # for YIBs file names
            model_name = model_name.replace('_Monthly','')

        if (
            var_name == 'nbp' and
            (model_name.startswith("DLEM")
            or model_name.startswith("LPJ-GUESS"))
        ):
            # notes
            # 1. DLEM does not have monthly output for nbp
            # 2. LPJ-GUESS does not have monthly output for nbp
            continue

        # read TRENDY output with correct timestamps
        ds = xr.open_dataset(f, decode_times=False)
        # detect the name of time coordinate
        time_name: str = [
            typing.cast(str, coord)
            for coord in ds.coords
            if typing.cast(str, coord).startswith("time")
        ][0]
        if time_name != "time":
            ds = ds.rename({time_name: "time"})

        if model_name.startswith("VISIT-NIES"):
            # VISIT-NIES time variable is in the unit of years since AD 0-Jan-1st
            _time_start_str: str = '1700-01-15'
        elif model_name.startswith("CLASSIC"):
            # CLASSIC start time: 1700-12-31
            _time_start_str: str = '1701-01-15'
        else:
            _time_start_str: str = ds.time.units.split("since")[1]
        print(f"* {model_name} _time_start_str: {_time_start_str}")

        # re-assign timestamps
        if  var_name in ['cLitter', 'cSoil'] and model_name.split("_")[0] in ['CLASSIC', 'JSBACH', 'JULES']: # annual resolution # ISBA-CTRIP has monthly output for carbon pools
            _time_start: pd.Timestamp = (
                pd.Timestamp(_time_start_str)
                + pd.offsets.YearEnd(0)
                + pd.offsets.YearBegin(-1)
            )
            print(f"* {model_name} modified _time_start_str: {_time_start}")

            _timestamps: pd.DatetimeIndex = pd.date_range(
                start=_time_start, periods=len(ds.time), freq="YS"
            )
            ds = ds.assign_coords(time=_timestamps)
            print(f"* {model_name} is read successfully")
        
        else: # monthly resolution
            _time_start: pd.Timestamp = (
                pd.Timestamp(_time_start_str)
                + pd.offsets.MonthEnd(0)
                + pd.offsets.MonthBegin(-1)
            )
            print(f"* {model_name} modified _time_start_str: {_time_start}")

            _timestamps: pd.DatetimeIndex = pd.date_range(
                start=_time_start, periods=len(ds.time), freq="MS"
            )
            ds = ds.assign_coords(time=_timestamps)
            print(f"* {model_name} is read successfully")

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

        #change zeros in the YIBs ouput to NaN
        if model_name.startswith("YIBs") and var_name in ["gpp", "nbp", "ra", "rh"]:
            _tmp = ds[var_name].values
            _tmp[_tmp == 0] = np.nan
            ds[var_name].values = _tmp

        # detect the original resolution
        res_lat: int = ds.latitude.values[1] - ds.latitude.values[0]
        res_lon: int = ds.longitude.values[1] - ds.longitude.values[0]

        # select the period for model benchmarking
        ds_subset: xr.Dataset = ds.isel(
            time=(ds.time.dt.year >= year_start)
            & (ds.time.dt.year <= year_end),
            latitude=(ds.latitude >= -90) & (ds.latitude <= 90),
        )

        # aggregate tsl by layer depth (top 10 cm)
        if var_name == 'tsl':
            ds_subset = aggrgate_tsl_by_depth (model_name, ds_subset)
            print('tsl aggregated by layer depth')


        if "ISBA-CTRIP" in model_name:
            ds_subset = ds_subset.drop_vars(["lon_FULL_bnds", "lat_FULL_bnds"])
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

        if (np.isclose(res_lat, res_out) and 
            np.isclose(res_lon, res_out) and 
            np.isclose(np.max(ds.longitude.values), 179.75) and
            np.isclose(np.min(ds.longitude.values), -179.75) and
            np.isclose(np.max(ds.latitude.values), 89.75) and
            np.isclose(np.min(ds.latitude.values), -89.75)):
            # no need to regrid if the grid is the same as grid_out (half-degree, lat [-90, 90], lon [-180, 180]
            # note CLM5.0, ISAM, JULES have longitude as [0, 360] - it seems that xesmf can automatically regrid them to [-180, 180]
            ds_subset_out: xr.Dataset = ds_subset
            print(f"* {model_name} no need for re-gridding")
        else:
            # regrid to half-degree resolution
            regridder = xe.Regridder(ds_subset, grid_out, "conservative")
            ds_subset_out = regridder(ds_subset)
            print(f"* {model_name} re-gridded successfully")

        # harmonize the timestamps: keep the days on the first of each month
        # note: this will alter non-standard calendars, for example, a calendar
        # without a leap day
        if  var_name in ['cLitter', 'cSoil'] and model_name.split("_")[0] in ['CLASSIC', 'JSBACH', 'JULES']: # annual resolution # ISBA-CTRIP has monthly output for carbon pools
            timestamps: pd.DatetimeIndex = pd.date_range(
                start=f"{year_start}-01-01", end=f"{year_end}-12-31", freq="YS"
            )
        else: # monthly resolution
            timestamps: pd.DatetimeIndex = pd.date_range(
                start=f"{year_start}-01-01", end=f"{year_end}-12-31", freq="MS"
            )

        ds_subset_out["time"] = timestamps.to_numpy()

        f_o: str = (
            f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/trendy-v11/global-half-degree/"
            f"{model_name}-half-degree.nc"
        )
        # mark ISBA-CTRIP carbon pool output as monthly
        if "ISBA-CTRIP" in model_name and var_name in ['cLitter', 'cSoil']:
            f_o = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/trendy-v11/global-half-degree/{model_name}-half-degree-monthly.nc"

        ds_subset_out.to_netcdf(
            f_o,
            engine="netcdf4",
            encoding={v: compression for v in ds_subset_out.data_vars},
        )

        del ds, ds_subset, ds_subset_out, f_o
        print(f"* {model_name} done")

def aggrgate_tsl_by_depth (model_name, ds):

    # for tsl, rename soil depth layer coordinate and select only ~ top 10cm
    if 'CLASSIC' in model_name:  # soil depth is specified in coordinates
        ds = ds.isel(stlayer=0).squeeze().drop_vars('stlayer') # squeeze function removes the dimension if its length is one # first layer 10 cm
    elif 'ISBA-CTRIP' in model_name:
        ds_tmp = ds
        ds = ds.isel(sdepth=0).drop_vars('sdepth_bnds').drop_vars('sdepth') # set the dimension
        ds['tsl'].values = (ds_tmp['tsl'].isel(sdepth=0)*0.01 + ds_tmp['tsl'].isel(sdepth=1)*0.03 + ds_tmp['tsl'].isel(sdepth=2)*0.06) / 0.1 # first three layers 1, 3, 6 cm thick
    elif 'JSBACH' in model_name:
        ds = ds.isel(depth=0).squeeze().drop_vars('depth') # first layer 6.5 cm
    elif 'JULES' in model_name:
        ds = ds.isel(layer=0).squeeze().drop_vars('layer') # first layer 10 cm
    elif 'CABLE-POP' in model_name: 
        ds = (ds.isel(soil=0) * 0.022 + ds.isel(soil=1) * 0.058) / 0.08 # first two layers are 2.2 and 5.8 cm thick

    return ds


if __name__ == "__main__":
    start_time: float = time.time()
    for var_name in ['cLitter', 'cSoil']: #"gpp", "nbp", "ra", "rh", "lai", 'tsl', 'cLitter', 'cSoil'
        regrid_trendy(var_name=var_name)

    end_time: float = time.time()
    print("Done. This took %.0f seconds" % (end_time - start_time))