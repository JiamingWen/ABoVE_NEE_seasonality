"""Extract background concentrations for CARVE and Arctic-CAP aircraft data.
Uses CarbonTracker 2022 data or empirical background data from NOAA GML."""

import glob
import os
import time
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy import stats
import multiprocessing

def _extract_endpts_co2_ct(
    traj: xr.Dataset,
    flist: List[str],
) -> pd.DataFrame:
    """Extract end-point CO2 concentrations from CarbonTracker 2022 data."""
    # note: this function is only called from within this code and it applies
    # only to a single trajectory file
    date_strs: List[str] = list(
        map(
            str,
            np.sort(
                np.unique(traj["endptsdate"].values.astype("datetime64[D]"))
            ),
        )
    ) # the endpoints may be located on several days in the past 5 or 10 days

    # find the CarbonTracker files of the corresponding dates
    flist_select: List[str] = [
        f
        for f in flist
        if os.path.basename(f).replace(".nc", "").split("_")[-1] in date_strs
    ]

    if len(flist_select) != len(date_strs):
        raise ValueError(
            f"Expected {len(date_strs)} CT CO2 files, but found {len(flist_select)}."
        )

    # combine the selected CarbonTracker files into a single dataset along the time axis
    ds_co2: xr.Dataset = xr.open_mfdataset(flist_select).compute()

    # create a dataframe for end points
    endpts_colnames: List[str] = list(
        map(lambda x: b"".join(x).decode("utf-8"), traj["endptsnames"].values)
    )
    df_endpts: pd.DataFrame = pd.DataFrame({"date": traj["endptsdate"].values})
    for i, s in enumerate(endpts_colnames):
        df_endpts[s] = traj["endpts"].values[i, :]

    # regularize data
    df_endpts["index"] = df_endpts["index"].astype("int")
    df_endpts["pres"] *= 1e2  # hPa -> Pa

    # extract the nearest CO2 mole fraction estimates in space and time
    df_endpts["co2_ct"] = np.nan

    # use longitudes in 0 to 360 for matching to avoid cutoff at the date line
    lons_360: npt.NDArray[np.float64] = ds_co2[
        "longitude"
    ].values.copy()  # must make a copy!
    lons_360[lons_360 < 0.0] += 360
    for i in df_endpts.index:
        endpt_date: np.datetime64 = np.datetime64(
            df_endpts.loc[i, "date"], "ns"
        )
        endpt_lat: float = df_endpts.loc[i, "lat"]
        endpt_lon: float = df_endpts.loc[i, "lon"]
        endpt_pres: float = df_endpts.loc[i, "pres"]
        i_date: int = np.argmin(
            np.abs(ds_co2["time"].values - endpt_date), keepdims=True
        )[0]
        i_lat: int = np.argmin(
            np.abs(ds_co2["latitude"].values - endpt_lat), keepdims=True
        )[0]

        if endpt_lon < 0:
            endpt_lon_360: float = endpt_lon + 360
        else:
            endpt_lon_360 = endpt_lon

        i_lon: int = np.argmin(
            np.abs(lons_360 - endpt_lon_360), keepdims=True
        )[0]

        pres_boundaries: np.ndarray = (
            ds_co2["pressure"]
            .isel(time=i_date, latitude=i_lat, longitude=i_lon)
            .values
        )
        i_pres: int = np.where((pres_boundaries - endpt_pres) < 0)[0][0] - 1

        if i_pres == -1:
            df_endpts.loc[i, "co2_ct"] = (
                ds_co2["pbl_co2"]
                .isel(time=i_date, latitude=i_lat, longitude=i_lon)
                .values.item()
            )
        else:
            df_endpts.loc[i, "co2_ct"] = (
                ds_co2["co2"]
                .isel(
                    time=i_date, latitude=i_lat, longitude=i_lon, level=i_pres
                )
                .values.item()
            )

    return df_endpts

def extract_endpts_co2_ct_airborne(
    fname_traj_list: str,
    traj_path: str,
    bg_path: str,
    dest_path: str,
) -> None:
    """Extract CO2 background concentrations from CarbonTracker 2022 for aircraft data."""
    df: pd.DataFrame = pd.read_csv(
        fname_traj_list, engine="c", parse_dates=["footprint_time_UTC"]
    )
    flist_co2_ct: List[str] = sorted(
        glob.glob(f"{bg_path}/CT2022.molefrac_glb3x2_*.nc")
    )

    '''Option 1: running in a loop'''
    for i in df.index:
        filename_trajectory: str = df.loc[i, "footprint_filename"].replace("foot", "stilt")
        path_trajectory: str = (
            f"{traj_path}/{filename_trajectory}"
        )
        if os.path.exists(path_trajectory):
            traj: xr.Dataset = xr.open_dataset(path_trajectory, decode_times=False)
            if 'endptsdate' in traj.variables: # Manually decode only the 'endptsdate' variable to datetime64, there is an issue with another time variable in certain files
                traj['endptsdate'] = xr.decode_cf(traj[['endptsdate']])['endptsdate']
            df_endpts: pd.DataFrame = _extract_endpts_co2_ct(
                traj, flist_co2_ct
            )
            # save end-point concentrations
            filename_endpts: str = filename_trajectory.replace(
                "stilt", "endpts"
            ).replace(".nc", ".csv")
            output_path: str = f"{dest_path}"
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            fo: str = f"{dest_path}/{filename_endpts}"
            df_endpts.to_csv(
                fo,
                index=False,
            )
            print(f"Saved end-point concentrations to {fo}")

#     '''Option 2: running in parallel'''
#     # ~ 100 files, 2 minutes to run, with 20-50 cores (does not run faster with more cores), similar to the single loop
#     df_selected_col = df[["footprint_filename"]].copy()
#     del df
#     num_workers = 55
#     args_list = [(i, df_selected_col, traj_path, flist_co2_ct, dest_path) for i in df_selected_col.index]
#     with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
#         pool.map(process_trajectory_parallel_ct, args_list)

# def process_trajectory_parallel_ct(args):
#     i, df_selected_col, traj_path, flist_co2_ct, dest_path = args
#     filename_trajectory: str = df_selected_col.loc[i, "footprint_filename"].replace("foot", "stilt")
#     path_trajectory: str = f"{traj_path}/{filename_trajectory}"
#     traj: xr.Dataset = xr.open_dataset(path_trajectory)
#     df_endpts: pd.DataFrame = _extract_endpts_co2_ct(traj, flist_co2_ct)
#     filename_endpts: str = filename_trajectory.replace("stilt", "endpts").replace(".nc", ".csv")
#     fo: str = f"{dest_path}/{filename_endpts}"
#     df_endpts.to_csv(fo, index=False)
#     print(f"Saved end-point concentrations to {fo}")

def calculate_background_co2_ct_airborne(
    fname_traj_list: str,
    endpts_path: str,
    dest: str,
) -> None:
    """Calculate background CO2 concentrations for aircraft data using CT2022 data."""
    df: pd.DataFrame = pd.read_csv(
        fname_traj_list,
        engine="c",
        parse_dates=["footprint_time_UTC"],
        usecols=[
            "footprint_filename",
            "footprint_time_UTC",
            "footprint_time_AKT",
            "footprint_lat",
            "footprint_lon",
            "footprint_agl",
            "airborne_CO2",
        ],
    )
    # add columns for background concentration statistics
    df["filename_endpts"] = ""
    df["co2_bg_ct"] = np.nan
    df["co2_bg_ct.sem"] = np.nan
    df["co2_bg_ct.sd"] = np.nan
    df["co2_bg_ct.n"] = -1
    df["co2_bg_ct.median"] = np.nan
    df["co2_bg_ct.q1"] = np.nan
    df["co2_bg_ct.q3"] = np.nan

    for i in df.index:
        filename_endpts: str = (
            df.loc[i, "footprint_filename"]
            .replace("foot", "endpts")
            .replace(".nc", ".csv")
        )
        path_endpts: str = (
            f"{endpts_path}/{filename_endpts}"
        )
        if os.path.exists(path_endpts):
            df.loc[i, "filename_endpts"] = os.path.basename(path_endpts)
            _df_endpts: pd.DataFrame = pd.read_csv(path_endpts)
            co2_bg_ct: npt.NDArray[np.float64] = _df_endpts["co2_ct"].values
            co2_bg_ct = co2_bg_ct[np.isfinite(co2_bg_ct)]  # remove NaNs
            df.loc[i, "co2_bg_ct"] = np.mean(co2_bg_ct)
            df.loc[i, "co2_bg_ct.sem"] = stats.sem(co2_bg_ct)
            df.loc[i, "co2_bg_ct.sd"] = np.std(co2_bg_ct, ddof=1)
            df.loc[i, "co2_bg_ct.n"] = len(co2_bg_ct)
            df.loc[i, "co2_bg_ct.median"] = np.median(co2_bg_ct)
            df.loc[i, "co2_bg_ct.q1"] = np.quantile(co2_bg_ct, 0.25)
            df.loc[i, "co2_bg_ct.q3"] = np.quantile(co2_bg_ct, 0.75)
            del _df_endpts

    df.to_csv(dest, index=False)
    print(f"Saved CO2 background concentrations to {dest}.")


def _extract_endpts_co2_ebg(
    traj: xr.Dataset,
    ebg: xr.Dataset,
) -> pd.DataFrame:
    """Extract end-point CO2 concentrations from empirical background."""
    # note: called from within this file and only applies to one trajectory

    # create a dataframe for end points
    endpts_colnames: List[str] = list(
        map(lambda x: b"".join(x).decode("utf-8"), traj["endptsnames"].values)
    )
    df_endpts: pd.DataFrame = pd.DataFrame({"date": traj["endptsdate"].values})
    for i, s in enumerate(endpts_colnames):
        df_endpts[s] = traj["endpts"].values[i, :]

    # regularize data
    df_endpts["index"] = df_endpts["index"].astype("int")
    df_endpts["pres"] *= 1e2  # hPa -> Pa
    df_endpts["alt"] = df_endpts["agl"] + df_endpts["grdht"]  # altitude [m]

    # extract the nearest CO2 mole fraction estimates in space and time
    df_endpts["co2_ebg"] = np.nan

    # use longitudes in 0 to 360 for matching to avoid cutoff at the date line
    lons_360: npt.NDArray[np.float64] = ebg[
        "lon"
    ].values.copy()  # must make a copy!
    lons_360[lons_360 < 0.0] += 360
    for i in df_endpts.index:
        endpt_date: np.datetime64 = np.datetime64(
            df_endpts.loc[i, "date"], "ns"
        )
        endpt_lat: float = df_endpts.loc[i, "lat"]
        endpt_lon: float = df_endpts.loc[i, "lon"]
        endpt_alt: float = df_endpts.loc[i, "alt"]
        i_date: int = np.argmin(
            np.abs(ebg["date"].values - endpt_date), keepdims=True
        )[0]
        i_lat: int = np.argmin(
            np.abs(ebg["lat"].values - endpt_lat), keepdims=True
        )[0]

        if endpt_lon < 0:
            endpt_lon_360: float = endpt_lon + 360
        else:
            endpt_lon_360 = endpt_lon

        i_lon: int = np.argmin(
            np.abs(lons_360 - endpt_lon_360), keepdims=True
        )[0]
        i_alt: int = np.argmin(
            np.abs(ebg["alt"].values - endpt_alt), keepdims=True
        )[0]
        df_endpts.loc[i, "co2_ebg"] = (
            ebg["value"]
            .isel(date=i_date, lat=i_lat, lon=i_lon, alt=i_alt)
            .values.item()
        )

    return df_endpts


def calculate_ebg_co2_airborne(
    fname_traj_list: str,
    traj_path: str,
    fname_ebg: str,
    dest: str,
) -> None:
    """Calculate empirical background CO2 concentrations for aircraft data."""
    # note: this function does not save intermediate output

    df: pd.DataFrame = pd.read_csv(
        fname_traj_list, engine="c", parse_dates=["footprint_time_UTC"]
    )

    ebg: xr.Dataset = xr.open_dataset(fname_ebg)

    # add columns for background concentration statistics
    df["filename_endpts"] = ""
    df["co2_ebg"] = np.nan
    df["co2_ebg.sem"] = np.nan
    df["co2_ebg.sd"] = np.nan
    df["co2_ebg.n"] = -1
    df["co2_ebg.median"] = np.nan
    df["co2_ebg.q1"] = np.nan
    df["co2_ebg.q3"] = np.nan

    for i in df.index:
        filename_trajectory: str = df.loc[i, "footprint_filename"].replace("foot", "stilt")
        path_trajectory: str = (
            f"{traj_path}/{filename_trajectory}"
        )
        if os.path.exists(path_trajectory):
            traj: xr.Dataset = xr.open_dataset(path_trajectory, decode_times=False)
            if 'endptsdate' in traj.variables: # Manually decode only the 'endptsdate' variable to datetime64, there is an issue with another time variable in certain files
                traj['endptsdate'] = xr.decode_cf(traj[['endptsdate']])['endptsdate']
            df_endpts: pd.DataFrame = _extract_endpts_co2_ebg(
                traj, ebg
            )
            co2_ebg: npt.NDArray[np.float64] = df_endpts["co2_ebg"].values
            co2_ebg = co2_ebg[np.isfinite(co2_ebg)]  # remove NaNs
            df.loc[i, "co2_ebg"] = np.mean(co2_ebg)
            df.loc[i, "co2_ebg.sem"] = stats.sem(co2_ebg)
            df.loc[i, "co2_ebg.sd"] = np.std(co2_ebg, ddof=1)
            df.loc[i, "co2_ebg.n"] = len(co2_ebg)
            df.loc[i, "co2_ebg.median"] = np.median(co2_ebg)
            df.loc[i, "co2_ebg.q1"] = np.quantile(co2_ebg, 0.25)
            df.loc[i, "co2_ebg.q3"] = np.quantile(co2_ebg, 0.75)

    df.to_csv(dest, index=False)
    print(f"Saved empirical CO2 background concentrations to {dest}.")


if __name__ == "__main__":

    start_time: float = time.time()

    for campaign_name, year in [("carve", 2012), ("carve", 2013), ("carve", 2014), ("arctic_cap", 2017)]:
        
        if campaign_name == "carve":
            traj_path = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/carve-trajectories/CARVE_L4_WRF-STILT_Particle/data/CARVE-{year}-aircraft-particle-files-convect"
        else:
            traj_path = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/above-trajectories/ABoVE_Particles_WRF_AK_NWCa/data/ArcticCAP_2017_insitu-particles"
        
        extract_endpts_co2_ct_airborne(
            f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_matching_footprint.csv",
            traj_path,
            f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/background/co2/carbontracker-2022",
            f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/endpts/{year}",
        )
        calculate_background_co2_ct_airborne(
            f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_matching_footprint.csv",
            f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/endpts/{year}",
            f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_background-ct.csv",
        )
        calculate_ebg_co2_airborne(
            f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_matching_footprint.csv",
            traj_path,
            f"/central/groups/carnegie_poc/michalak-lab/data/noaa-gml-na-boundary-conditions/v20200302/nc/ebg_co2.nc",
            f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_background-ebg.csv",
        )
    end_time: float = time.time()
    print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
