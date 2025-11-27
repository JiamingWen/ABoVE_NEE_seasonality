'''download hourly X-BASE outputs'''
'''reference: https://gitlab.gwdg.de/fluxcom/fluxcomxdata/-/blob/main/docs/02-full-resolution.md'''

import fsspec
import xarray as xr
import calendar

variable = "NEE"

m = fsspec.get_mapper('https://swift.dkrz.de/v1/dkrz_a1e106384d7946408b9724b59858a536/fluxcom-x/FLUXCOMxBase/{}/'.format(variable))
ds = xr.open_zarr(m)

# subselect 2012-2014 and 2017 growing seasons, save file for every date
for year in [2012, 2013, 2014, 2017]:
    for month in range(4, 12):
        last_day = calendar.monthrange(year, month)[1]
        for day in range(1, last_day + 1):
            print(f"Processing {year}-{month:02d}-{day:02d}")
            date_str = f"{year}-{month:02d}-{day:02d}"
            ds_day = ds[['NEE']].sel(time=slice(date_str, date_str), lat=slice(90, 30))
            encoding = {
                'NEE': {
                    'zlib': True,
                    'complevel': 4,
                }
            }
            ds_day.to_netcdf(
                f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/x_base_hourly/X-BASE_NEE_hourly_{year}_{month:02d}_{day:02d}.nc',
                encoding=encoding,
            )
            print(f"Saved X-BASE_NEE_hourly_{year}_{month:02d}_{day:02d}.nc")
