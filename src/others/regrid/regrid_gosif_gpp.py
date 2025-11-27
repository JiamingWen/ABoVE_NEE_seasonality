"""Regridding GOSIF GPP to half-degree"""

import glob
from typing import List
import rioxarray
import xarray as xr
import pandas as pd
import os
import xesmf as xe
import numpy as np
import time

def regrid_gosif_gpp(year, geotiff_list) -> None:
    
    # combine tiff files to a single xarray dataset
    time_var = xr.Variable('time', [extract_time(f) for f in geotiff_list])

    # Load in and concatenate all individual GeoTIFFs
    geotiffs_da = xr.concat([rioxarray.open_rasterio(i) for i in geotiff_list],
                            dim=time_var)

    # Covert xarray.DataArray into a xarray.Dataset
    geotiffs_ds = geotiffs_da.to_dataset('band')
    del time_var, geotiffs_da

    # Rename the variable to a more useful name
    geotiffs_ds = geotiffs_ds.rename({1: 'GOSIF_GPP'})

    # Set fill values 65535 (water bodies) and 65534 (lands under snow/ice throughout the year)
    geotiffs_ds = geotiffs_ds.where(geotiffs_ds['GOSIF_GPP'] <= 65533)  

    # Apply scaling factor 0.01
    scaling_factor = 0.01
    ds = geotiffs_ds * scaling_factor # unit: g C m-2 mo-1

    # change nan to zero - otherwise the value at coarse grids will be zero if any nan value falls into it
    ds = ds.fillna(0)

    print('combined into a single xarray dataset, now doing the regridding')
    
    """Regrid GOSIF GPP."""
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

    print(f"Regridding GOSIF GPP data to monthly, half-degree resolution:")

    # change coordinate names
    ds = ds.rename(
        {"y": "latitude", "x": "longitude"}
    )
    
    # select the period for model benchmarking
    ds_subset = ds

    regridder = xe.Regridder(ds_subset, grid_out, "conservative")
    ds_subset_out = regridder(ds_subset)

    # export
    f_o: str = (
        f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gosif-gpp/global-half-degree/gosif-gpp-half-degree-{year}.nc"
    )
    ds_subset_out.to_netcdf(
        f_o,
        engine="netcdf4",
        encoding={v: compression for v in ds_subset_out.data_vars},
    )

    del ds, ds_subset_out, f_o


# extract year
def extract_year(f: str):
    tmp = os.path.basename(f)
    year = int(tmp[10:14])
    return year


# extract time
def extract_time(f: str):
    tmp = os.path.basename(f)
    yearstr = tmp[10:14]
    monthstr = tmp[16:18]
    time_str = pd.Timestamp(f"{yearstr}-{monthstr}-01")
    return time_str

# regrid
if __name__ == "__main__":
    
    start_time: float = time.time()

    input_dir = '/central/groups/carnegie_poc/michalak-lab/data/gosif-gpp/data.globalecology.unh.edu/data/GOSIF-GPP_v2/Monthly/Mean/'

    # read filenames into a list
    geotiff_list: List[str] = sorted(
        glob.glob(f"{input_dir}/GOSIF_GPP_*_Mean.tif")
    )

    geotiff_year = [extract_year(f) for f in geotiff_list]

    for year in np.arange(2000, 2024): # 2000
        print(year)
        geotiff_list_subset = [geotiff_list[index] for (index, value) in enumerate(geotiff_year) if value == year]
        regrid_gosif_gpp(year, geotiff_list_subset)

    end_time: float = time.time()
    print("Done. This took %.0f seconds" % (end_time - start_time))