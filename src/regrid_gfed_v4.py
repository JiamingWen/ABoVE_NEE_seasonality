"""Regrid GFEDv4 fire emissions."""
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import time
import h5py

def regrid_gfed(year: int) -> None:
    """Regrid GFEDv4 fire emissions."""
    input_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/gfed/v4.1s/"
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

    if (year <= 2016):
        filename = input_dir + f"GFED4.1s_{year}.hdf5"
    else:
        filename = input_dir + f"GFED4.1s_{year}_beta.hdf5"

    emission_factor = pd.read_table('/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gfed/v4/GFED4_Emission_Factors.txt', 
                                    skiprows=17, header=None, nrows=3, delim_whitespace=True, 
                                    names=['C_species','AGRI', 'BORF', 'DEFO', 'PEAT', 'SAVA', 'TEMF'])
    
    f = h5py.File(filename)

    # lat = f['lat'][()]
    # lon = f['lon'][()]

    CO2_emission_year = np.zeros(shape=(720,1440,12))

    for month in np.arange(1,13):
        print(month)
        monthstr = str(month).zfill(2)

        DM_emission = f['emissions'][monthstr]['DM'][()] #unit kg DM m-2 month-1

        #derive CO2 emissions from DM emissions, fraction of DM emissions from different land covers, and emission factors
        CO2_emission_month = np.zeros(shape=DM_emission.shape)

        for LCname in ['AGRI', 'BORF', 'DEFO', 'PEAT', 'SAVA', 'TEMF']:
            LC_DM_frac = f['emissions'][monthstr]['partitioning']['DM_'+LCname][()]
            LC_DM_emission = DM_emission * LC_DM_frac
            LC_CO2_emission = LC_DM_emission * float(emission_factor[LCname][emission_factor['C_species']=='CO2'])
            CO2_emission_month += LC_CO2_emission
        
        CO2_emission_year[:,:,month-1] = CO2_emission_month

    timestamps: pd.DatetimeIndex = pd.date_range(
        start=f"{year}-01-01", end=f"{year}-12-31", freq="MS"
    ).to_numpy()

    longitude = np.linspace(-180.0 + 0.5 * 0.25, 180.0 - 0.5 * 0.25, round(360.0 / 0.25))
    latitude = np.linspace(90.0 - 0.5 * 0.25, -90.0 + 0.5 * 0.25, round(180.0 / 0.25))

    ds = xr.Dataset(
        data_vars=dict(
            CO2_emission=(["latitude", "longitude", "time"], CO2_emission_year),
            ),
        coords=dict(
            longitude=(["longitude"], longitude),
            latitude=(["latitude"], latitude),
            time=timestamps,
        ),
    )

    regridder = xe.Regridder(ds, grid_out, "conservative")
    ds_out = regridder(ds)
    print(f"* {year} re-gridded successfully")

    # set the time variable
    timestamps: pd.DatetimeIndex = pd.date_range(
    start=f"{year}-01-01", end=f"{year}-12-31", freq="MS"
    )
    ds_out["time"] = timestamps.to_numpy()

    f_o: str = (
        f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gfed/v4.1s/global-half-degree/"
        f"GFED4.1s-half-degree-{year}.nc"
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
    year_start: int = 1997
    year_end: int = 2023
    for year in np.arange(year_start, year_end+1):
        regrid_gfed(year=year)

    end_time: float = time.time()
    print("Done. This took %.0f seconds" % (end_time - start_time))