import os
import pandas as pd
import numpy as np
import xarray as xr
import pytz
import datetime

year = 2012 # 2012 2013 2014

airborne_df = pd.read_csv(f'/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/carve-airborne/CARVE_L2_AtmosGas_Merge_1402/data/carve_AtmosISGA_L2_Merge_{year}_20160722.csv')
airborne_df.replace({'CO2.X': -999.9, 'CO.X': -999.9}, np.nan, inplace=True)
footprint_path = f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/carve/CARVE_L4_WRF-STILT_Footprint/data/CARVE-{year}-aircraft-footprints-convect/"
footprint_list = sorted(os.listdir(footprint_path))


# extract footprint information
footprint_info = pd.DataFrame()
for footprint_file in footprint_list:
    print(footprint_file)

    #footprint data
    with xr.open_dataset(filename_or_obj  = footprint_path+footprint_file) as footprint_data:
        footprint_lat = float(footprint_data.origlat) # same as in the file name, but in higher precision
        footprint_lon = float(footprint_data.origlon)
        footprint_agl = int(footprint_data.origagl)
        timestr = ''.join([x.decode('ASCII') for x in footprint_data.origutctime.values[0]])
        if timestr[17:19] == '60': #if second is 60, replace with 59
            timestr = timestr[0:17] + '59' 
        footprint_time_UTC = pd.to_datetime(timestr)

    single_record = pd.DataFrame({'footprint_filename': footprint_file,
                                  'footprint_time_UTC': footprint_time_UTC, 
                                  'footprint_lat': footprint_lat,
                                  'footprint_lon': footprint_lon, 
                                  'footprint_agl': footprint_agl}, index=[0])
    footprint_info = pd.concat([footprint_info, single_record])

# sort the files in time order (some files collected in the same minute may not be in the right order)
start_time = pd.to_datetime(footprint_info['footprint_time_UTC'][0]) # first timestamp
time_diff_tmp = footprint_info['footprint_time_UTC']
time_diff = (pd.to_datetime(time_diff_tmp) - start_time).dt.total_seconds()
footprint_info['time_diff'] = time_diff
footprint_info_sorted = footprint_info.sort_values(['footprint_time_UTC'], ascending=[True])
footprint_info_sorted.to_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/carve_airborne/atm_obs/ABoVE_{year}_carve_airborne_footprint_info.csv', encoding='utf-8', index=False)

# combine footprints with corresponding airborne measurements
# the matching between airborne measurements and footprints is not as clear as for Arctic-CAP data
# Previous studies said "the data were aggregated horizontally into 5 km bins and vertically into 50 m bins below 1000 m above sea level (asl) and 100 m bins above 1000 m asl"
# but it is not clear how the bins were defined
# so I defined my matching rule:
# average airborne measurements within a time interval before and after the footprint measurement
# the time interval is determined by the intervals between the footprint measurements, from seconds to 2 minutes, so that there are roughly equal number of observations for aggregation before and after the footprint time
# 2 minutes is a threshold I set to avoid averaging too many airborne measurements that are spatially far away from the location the footprint is generated for
# the mismatch in lat/lon/time of footprints and aggregated observations is very small

combined_df = pd.DataFrame()
airborne_startdate = datetime.datetime.strptime(f'{year}0101', "%Y%m%d")
start_time = pd.to_datetime(airborne_df['SOY'][0], unit='s', origin=airborne_startdate) # the first airborne measurement time
    
for footprint_num in np.arange(footprint_info_sorted.shape[0]): # iterate through all footprint files

    footprint_file = footprint_info_sorted['footprint_filename'].iloc[footprint_num]
    print(footprint_file)

    #footprint data
    footprint_lat = footprint_info_sorted['footprint_lat'].iloc[footprint_num]
    footprint_lon = footprint_info_sorted['footprint_lon'].iloc[footprint_num]
    footprint_agl = footprint_info_sorted['footprint_agl'].iloc[footprint_num]
    footprint_time_UTC = pd.to_datetime(footprint_info_sorted['footprint_time_UTC'].iloc[footprint_num])
    footprint_time_AKT = footprint_time_UTC.tz_localize('UTC').astimezone(pytz.timezone('America/Anchorage'))  #GMT-8; 
    # pytz already considers the daylight saving period. If the time is in April, it uses GMT-8. If the time is in January, it uses GMT-9.
    print(footprint_time_UTC)
    # calculate the time difference between the first airborne measurement and the first footprint, use it as half of the time interval for the first footprint
    time_interval_half = (footprint_time_UTC - start_time).total_seconds() 
    
    # if the half time interval is too small or too large, use a fixed time interval (10 seconds)
    if (time_interval_half < 0) or  (time_interval_half > 60): 
        # some files have very close timestamps; 
        # some files have very isolated timestamps; 
        
        time_interval_half = 5 # if this happens, use 10 seconds as time_interval to average nearby measurements
        # time_interval_half = time_interval/2 # use previous time_interval

        start_time = footprint_time_UTC - datetime.timedelta(seconds=time_interval_half)
        start_time = start_time.replace(second=int(np.floor(start_time.second/5)*5))
        start_time = start_time.replace(microsecond=0)

    time_interval = round(time_interval_half*2/5) * 5 # round to 5 sec
    end_time = start_time + datetime.timedelta(seconds=time_interval)
    print(start_time, end_time)

    # starting id to search in the airborne data
    if (footprint_num == 0):
        airborne_num_start = 0
    else:
        tmp = np.where(airborne_df['SOY'] == int((start_time - airborne_startdate).total_seconds()))[0]
        if len(tmp) > 0:
            airborne_num_start = tmp[0] # otherwise use the previous value

    # find corresponding airborne data
    airborne_num_list = []
    time_diff_sum = 0
    for i in np.arange(0, 200):
        airborne_num = airborne_num_start + i

        if airborne_num >= airborne_df.shape[0]:
            break

        airborne_startdate = datetime.datetime.strptime(f'{year}0101', "%Y%m%d")
        airborne_time = pd.to_datetime(airborne_df['SOY'][airborne_num], unit='s', origin=airborne_startdate)
        print(airborne_num, airborne_time)
        time_diff_start = int((airborne_time - start_time).total_seconds())
        time_diff_end = int((airborne_time - end_time).total_seconds())

        if (time_diff_start >= 0) and (time_diff_end <= 0):
            airborne_num_list.append(airborne_num)
            time_diff = int((airborne_time - footprint_time_UTC).total_seconds())
            time_diff_sum += time_diff
        
        if time_diff_end == 0:
            break
    
    if (len(airborne_num_list) > 0) and (abs(time_diff_sum)<60):
        airborne_time_diff = time_diff_sum / len(airborne_num_list)
        airborne_lat = np.mean(airborne_df['Lat'].loc[airborne_num_list])
        airborne_lon = np.mean(airborne_df['Long'].loc[airborne_num_list])
        airborne_alt = np.mean(airborne_df['GPS_Alt'].loc[airborne_num_list])
        lat_diff = airborne_lat - footprint_lat
        lon_diff = airborne_lon - footprint_lon
        print(airborne_time_diff, lat_diff, lon_diff, len(airborne_num_list))

        airborne_CO2 = np.nanmean(airborne_df['CO2.X'].loc[airborne_num_list])
        airborne_CO = np.nanmean(airborne_df['CO.X'].loc[airborne_num_list])
        airborne_start_id = np.min(airborne_num_list)
        airborne_n = len(airborne_num_list)

    else:
        airborne_time_diff = np.nan
        airborne_start_id = np.nan
        airborne_start_time = np.nan
        airborne_end_time = np.nan
        airborne_n = np.nan
        airborne_lat = np.nan
        airborne_lon = np.nan
        airborne_alt = np.nan
        airborne_CO2 = np.nan
        airborne_CO = np.nan

    single_record = pd.DataFrame({'footprint_filename': footprint_file,
                                'footprint_time_UTC': footprint_time_UTC, 
                                'footprint_time_AKT': footprint_time_AKT,
                                'footprint_lat': footprint_lat,
                                'footprint_lon': footprint_lon, 
                                'footprint_agl': footprint_agl,
                                'airborne_time_diff': airborne_time_diff,
                                'airborne_start_id': airborne_start_id,
                                'airborne_start_time': start_time,
                                'airborne_end_time': end_time,
                                'airborne_n': airborne_n,
                                'airborne_lat': airborne_lat,
                                'airborne_lon': airborne_lon,
                                'airborne_alt': airborne_alt,
                                'airborne_CO2': airborne_CO2,
                                'airborne_CO': airborne_CO}, index=[0])
    combined_df = pd.concat([combined_df, single_record])

    start_time = end_time + datetime.timedelta(seconds=5)

combined_df.to_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/carve_airborne/atm_obs/ABoVE_{year}_carve_airborne_matching_footprint.csv', encoding='utf-8', index=False)

# #check the coordinate matching
# np.sum(abs(combined_df.footprint_lat - combined_df.airborne_lat) > 0.0005)
# np.sum(abs(combined_df.footprint_lon - combined_df.airborne_lon) > 0.0005)
# (combined_df.footprint_lat - combined_df.airborne_lat).describe()
# (combined_df.footprint_lon - combined_df.airborne_lon).describe()
# combined_df['airborne_time_diff'].describe()

# combined_df[abs(combined_df.footprint_lat - combined_df.airborne_lat) > 0.01]