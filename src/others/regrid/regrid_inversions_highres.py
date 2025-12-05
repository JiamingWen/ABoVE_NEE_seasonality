'''regrid 3-hourly or daily inversions to 0.5 degree'''

import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import time
import rioxarray
import pyproj
from pyproj import CRS
import os

def regrid_inversion (inversion_name, input_dir, filename, varname, output_dir, regridder) -> None:
    
    '''Regrid inversions'''
    ds = xr.open_dataset(input_dir + filename, decode_coords="all")
    
    if inversion_name == "CTE":
        ds = ds.rename({"date": "time"})

    ds = ds[varname]
    ds_out = regridder(ds)
    print(f"* {filename} re-gridded successfully")
    
    if inversion_name == "CT-NOAA":
        outfilename = filename.replace('1x1', '0.5x0.5')
    else:
        outfilename = filename
        
    f_o: str = output_dir + outfilename
    
    compression = dict(zlib=True, complevel=5)
    ds_out.to_netcdf(
        f_o,
        engine="netcdf4",
        encoding={v: compression for v in ds_out.data_vars},
    )

    del ds, ds_out

start_year: int = 2012
end_year: int = 2019
inversion_name = "CTE" # CT-NOAA, CTE

if inversion_name == "CT-NOAA":
    input_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/original/CT-NOAA/three-hourly/"
    output_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/CT-NOAA/three-hourly/"
    regridder_dir: str = "/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_magnitude/data/CT-NOAA/"
    varname: str = ["bio_flux_opt"]
    yearstr_loc = list(range(15, 19))
elif inversion_name == "CTE":
    input_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/original/CTE/three-hourly/cte2024_posterior/co2_bio_flux_opt/"
    output_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/CTE/three-hourly/"
    regridder_dir: str = "/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_magnitude/data/CTE/"
    varname: str = ["co2_bio_flux_opt"]
    yearstr_loc = list(range(4, 8))

if __name__ == "__main__":

    start_time: float = time.time()

    '''grid for output data'''
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


    '''generate the regridder once and re-use it'''
    nc_files = [
        f for f in os.listdir(input_dir) 
        if f.endswith(".nc") and start_year <= int(f[yearstr_loc[0]:yearstr_loc[-1] + 1]) <= end_year
    ]

    f = input_dir + nc_files[0]

    ds = xr.open_dataset(f, decode_coords="all")
    ds = ds[varname]

    # # save regridder
    regridder = xe.Regridder(ds, grid_out, "conservative")
    fn = regridder.to_netcdf(f'{regridder_dir}regridder.nc')

    # # read saved regridder
    # fn = xr.open_dataset(f'{regridder_dir}regridder.nc')
    # regridder = xe.Regridder(ds, grid_out, 'conservative', weights=fn)


    '''loop through all files'''
    for nc_file in nc_files:
        regrid_inversion(inversion_name, input_dir, nc_file, varname, output_dir, regridder)

    end_time: float = time.time()
    print("Done. This took %.0f seconds" % (end_time - start_time))