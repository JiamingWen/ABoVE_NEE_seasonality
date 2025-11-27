# check above arctic-cap footprint data
import datetime
import xarray as xr

fn = '/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/above/ABoVE_Footprints_WRF_AK_NWCa/data/ArcticCAP_2017_insitu-footprints/foot2017x04x26x19x25x50.0227Nx104.1285Wx03012.nc'
fn = '/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/carve/CARVE_L4_WRF-STILT_Footprint/data/CARVE-2013-aircraft-footprints-convect/foot2013x04x02x23x21x64.8920Nx147.7411Wx00440.nc'

data = xr.open_dataset(filename_or_obj  = fn)

origlat = data.origlat # same as what in the file name
origlon = data.origlon
origagl = data.origagl
origutctime = data.origutctime

############################################################
footprint_file = 'foot2017x04x26x19x25x50.0227Nx104.1285Wx03012.nc'
footprint_year = int(footprint_file[4:8])
footprint_month = int(footprint_file[9:11])
footprint_date = int(footprint_file[12:14])
footprint_hour = int(footprint_file[15:17])
footprint_minute = int(footprint_file[18:20])
footprint_time = datetime.datetime(footprint_year, footprint_month, footprint_date, footprint_hour, footprint_minute)

footprint_lat = float(footprint_file[21:28])
footprint_lon = -float(footprint_file[30:38]) #add a minus sign for west lon

footprint_agl = int(footprint_file[40:45])
