'''check empirical background netcdf'''

import os
import xarray as xr
import pandas as pd
import numpy as np

ebg_filename = "/central/groups/carnegie_poc/michalak-lab/data/noaa-gml-na-boundary-conditions/v20200302/nc/ebg_co2.nc"
ds = xr.open_dataset(ebg_filename)
print(ds)