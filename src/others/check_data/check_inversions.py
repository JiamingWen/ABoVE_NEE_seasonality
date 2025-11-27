'''check inversions'''

import xarray as xr
import numpy as np
import typing
import pandas as pd
import xesmf as xe
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

ds = xr.open_dataset('/central/groups/carnegie_poc/michalak-lab/data/inversions/inversions_raw/GCP2023_inversions_1x1_version1_1_20240124.nc')
inversion_num = 5
inversion_name = ''.join(ds.ensemble_member_name[inversion_num].values)
print(inversion_name)

land_flux_prior = ds.prior_flux_land[inversion_num]
land_flux_posterior = ds.land_flux_only_fossil_cement_adjusted[inversion_num] #unit: PgC/m2/yr
ff = ds.fossil_fuel_emissions[inversion_num]

# Select a timestamp to plot
timestamp = '2017-01-01'

# Extract data for the selected timestamp
prior_flux = land_flux_prior.sel(time=timestamp)
posterior_flux = land_flux_posterior.sel(time=timestamp)
ff_flux = ff.sel(time=timestamp)

# Plot prior flux
plt.figure(figsize=(12, 6))
prior_flux.plot()
plt.title(f'Prior Land Flux on {timestamp}')
plt.show()

# Plot posterior flux
plt.figure(figsize=(12, 6))
posterior_flux.plot()
plt.title(f'Posterior Land Flux on {timestamp}')
plt.show()

# Plot FF emissions
plt.figure(figsize=(12, 6))
ff_flux.plot()
plt.title(f'FF emissions on {timestamp}')
plt.show()


# Check for non-NaN values in land_flux_prior
nonnan_values = land_flux_prior.notnull()
print(f'Number of non-NaN values in land_flux_prior: {nonnan_values.sum().item()}')

np.sum(ff_flux)