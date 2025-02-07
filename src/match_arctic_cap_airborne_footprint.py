'''combine ABoVE arctic-cap airborne and footprint data'''
import os
import pandas as pd
import numpy as np
import xarray as xr
import pytz

#airborne data
fn = '/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/arctic-cap/ABoVE_Arctic_CAP_1658/data/ABoVE_2017_insitu_10sec.nc'
airborne_data = xr.open_dataset(filename_or_obj  = fn)
airborne_df = pd.DataFrame({'airborne_time_UTC': airborne_data.time, 
                            'airborne_latitude': airborne_data.latitude, 
                            'airborne_longitude': airborne_data.longitude, 
                            'airborne_altitude': airborne_data.altitude,
                            'airborne_CO2': airborne_data.CO2, 
                            'airborne_CO2_nvalue': airborne_data.CO2_nvalue, 
                            'airborne_CO2_stdv': airborne_data.CO2_stdv, 
                            'airborne_CO2_unc': airborne_data.CO2_unc,
                            'airborne_CO': airborne_data.CO})
# Save airborne data to CSV
airborne_df.to_csv('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_10sec.csv', encoding='utf-8', index=False)


#read footprint data (20s intervals) and combine with airborne data (10s intervals)
footprint_path = "/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/above/ABoVE_Footprints_WRF_AK_NWCa/data/ArcticCAP_2017_insitu-footprints/"
footprint_list = sorted(os.listdir(footprint_path))

combined_df = pd.DataFrame([])
for footprint_file in footprint_list:
    print(footprint_file)

    #footprint data
    footprint_data = xr.open_dataset(filename_or_obj  = footprint_path+footprint_file)
    footprint_lat = float(footprint_data.origlat) # same as in the file name
    footprint_lon = float(footprint_data.origlon)
    footprint_agl = int(footprint_data.origagl)
    timestr = ''.join([x.decode('ASCII') for x in footprint_data.origutctime.values[0]])
    footprint_time_UTC = pd.to_datetime(timestr)
    footprint_time_AKT = footprint_time_UTC.tz_localize('UTC').astimezone(pytz.timezone('America/Anchorage'))  #GMT-8; 
    # pytz already considers the daylight saving period. If the time is in April, it uses GMT-8. If the time is in January, it uses GMT-9.

    #corresponding airborne data
    time_diff = (footprint_time_UTC - airborne_df.airborne_time_UTC).dt.total_seconds()
    id1 = time_diff[time_diff == 10].index.values
    id2 = time_diff[time_diff == 0].index.values

    if len(id1) == 1:
        airborne_df1 = airborne_df[time_diff == 10].copy()
    else:
        airborne_df1 = airborne_df[0:1].copy()
        airborne_df1.iloc[0] = np.nan

    if len(id2) == 1:
        airborne_df2 = airborne_df[time_diff == 0].copy()
    else:
        airborne_df2 = airborne_df[0:1].copy()
        airborne_df2.iloc[0] = np.nan
    
    airborne_time_UTC_1 = airborne_df1.airborne_time_UTC.values
    airborne_time_UTC_2 = airborne_df2.airborne_time_UTC.values

    airborne_lat_1 = airborne_df1.airborne_latitude.values
    airborne_lat_2 = airborne_df2.airborne_latitude.values
    airborne_lat = np.nanmean([airborne_lat_1, airborne_lat_2])

    airborne_lon_1 = airborne_df1.airborne_longitude.values
    airborne_lon_2 = airborne_df2.airborne_longitude.values
    airborne_lon = np.nanmean([airborne_lon_1, airborne_lon_2])

    airborne_alt_1 = airborne_df1.airborne_altitude.values
    airborne_alt_2 = airborne_df2.airborne_altitude.values
    airborne_alt = np.nanmean([airborne_alt_1, airborne_alt_2])

    airborne_CO2_1 = airborne_df1.airborne_CO2.values
    airborne_CO2_2 = airborne_df2.airborne_CO2.values
    airborne_CO2 = np.nanmean([airborne_CO2_1, airborne_CO2_2])

    airborne_CO_1 = airborne_df1.airborne_CO.values
    airborne_CO_2 = airborne_df2.airborne_CO.values
    airborne_CO = np.nanmean([airborne_CO_1, airborne_CO_2])

    single_record = pd.DataFrame({'footprint_filename': footprint_file,
                                  'footprint_time_UTC': footprint_time_UTC, 
                                  'footprint_time_AKT': footprint_time_AKT,
                                  'footprint_lat': footprint_lat,
                                  'footprint_lon': footprint_lon, 
                                  'footprint_agl': footprint_agl,
                                  'airborne_time_UTC_1': airborne_time_UTC_1,
                                  'airborne_time_UTC_2': airborne_time_UTC_2,
                                  'airborne_lat_1': airborne_lat_1,
                                  'airborne_lat_2': airborne_lat_2,
                                  'airborne_lat': airborne_lat,
                                  'airborne_lon_1': airborne_lon_1,
                                  'airborne_lon_2': airborne_lon_2,
                                  'airborne_lon': airborne_lon,
                                  'airborne_alt_1': airborne_alt_1,
                                  'airborne_alt_2': airborne_alt_2,
                                  'airborne_alt': airborne_alt,
                                  'airborne_CO2_1': airborne_CO2_1,
                                  'airborne_CO2_2': airborne_CO2_2,
                                  'airborne_CO2': airborne_CO2,
                                  'airborne_CO': airborne_CO}, index=[0])
    combined_df = pd.concat([combined_df, single_record])

# Save combined data to CSV
combined_df.to_csv('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_matching_footprint.csv', encoding='utf-8', index=False)

# # #check the coordinate matching
# np.sum(abs(combined_df.footprint_lat - combined_df.airborne_lat) > 0.0005)
# np.sum(abs(combined_df.footprint_lon - combined_df.airborne_lon) > 0.0005)

