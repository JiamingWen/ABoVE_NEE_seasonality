'''aggregate NEE from 3-hourly to daily and monthly, or daily to monthly'''

import xarray as xr
import os

start_year: int = 2012
end_year: int = 2019
inversion_name = "CTE" # CT-NOAA, CTE

if inversion_name == "CT-NOAA":
    varname: str = ["bio_flux_opt"]
    filestr = "CT2022.flux0.5x0.5."
elif inversion_name == "CTE":
    dir_3hourly: str = "/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/CTE/three-hourly/"
    varname: str = ["co2_bio_flux_opt"]
    filestr = "bio_"


dir_3hourly = f'/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/{inversion_name}/three-hourly/'
dir_daily = f'/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/{inversion_name}/daily/'
dir_monthly = f'/central/groups/carnegie_poc/michalak-lab/data/inversions/inversion_dif_tem_res/global-half-degree/{inversion_name}/monthly/'

for year in range(2012, 2020):
    for month in range(1, 13):
        print(f"Processing year: {year}, month: {month}")

        month_str = f"{month:02d}"

        if inversion_name in ["CT-NOAA", "CTE"]: # 3-hourly
            files = os.listdir(dir_3hourly)
            files_3hourly = [f for f in files if f.startswith(f'{filestr}{year}{month_str}')]

            ds_daily_list = []
            for file_name in files_3hourly:
                file_path = os.path.join(dir_3hourly, file_name)
                ds_3hourly = xr.open_dataset(file_path)
                ds_daily = ds_3hourly[varname].mean(dim='time')
                
                output_file = os.path.join(dir_daily, file_name)
                ds_daily.to_netcdf(output_file)

                ds_daily_list.append(ds_daily)

        else: # daily
            files = os.listdir(dir_daily)
            files_daily = [f for f in files if f.startswith(f'{filestr}{year}{month_str}')]
            
            ds_daily_list = []
            for file_name in files_daily:
                file_path = os.path.join(dir_daily, file_name)
                ds = xr.open_dataset(file_path)
                ds_daily = ds[varname]
                ds_daily_list.append(ds_daily)

        ds_monthly = xr.concat(ds_daily_list, dim='file').mean(dim='file')
        output_file = os.path.join(dir_monthly, f'{filestr}{year}{month_str}.nc')
        ds_monthly.to_netcdf(output_file)