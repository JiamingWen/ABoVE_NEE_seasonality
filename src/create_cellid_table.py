import numpy as np
import pandas as pd
import xarray as xr

lat = np.arange(30.25, 90, 0.5)
lon = np.arange(-179.75, 180, 0.5)
tmp = np.meshgrid(np.arange(len(lon)), np.arange(len(lat)))
lat_id = tmp[1].flatten().astype('i')
lon_id = tmp[0].flatten().astype('i')
cell_id = np.arange(len(lat)*len(lon)).astype('i')
coor_df = pd.DataFrame({'cell_id': cell_id, 'lat_id': lat_id, 'lon_id': lon_id, 
                        'lat': lat[lat_id], 'lon': lon[lon_id]})

# marking ocean/land pixels
ocean_mask = xr.open_dataset('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/above_mask/ocean-mask-half-degree.nc')
ocean_mask_subset: xr.Dataset = ocean_mask.isel(
    latitude=(ocean_mask.latitude >= 30) & (ocean_mask.latitude <= 90),
    )
value = ocean_mask_subset['seamask'].values #360 lat x 720 lon; land: 0, ocean: 1
latitude = ocean_mask_subset['latitude'].values
longitude = ocean_mask_subset['longitude'].values
longitude = (longitude + 180) % 360 - 180 
land_lat = latitude[np.where(value == 0)[0]]
land_lon = longitude[np.where(value == 0)[1]]

land_mask = np.zeros(coor_df.shape[0])
for i in np.arange(len(land_lat)):
    land_mask[np.where((coor_df['lat']==land_lat[i]) & (coor_df['lon']==land_lon[i]))] = 1 # land: 1; ocean:0

coor_df['land'] = land_mask.astype(int)


# marking ABoVE regions
ABoVE_mask = xr.open_dataset('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/above_mask/above_ext.nc')
ABoVE_mask = ABoVE_mask.rename({'lat': 'latitude', 'lon': 'longitude'})
ABoVE_mask_subset: xr.Dataset = ABoVE_mask.isel(
    latitude=(ABoVE_mask.latitude >= 30) & (ABoVE_mask.latitude <= 90),
    )
value = ABoVE_mask_subset['ids'].values # ABoVE core region: 0; ABoVE extended region: 1; outside of ABoVE: 255
coor_df['ABoVE'] = value.flatten().astype(int)


# marking land covers
lc = xr.open_dataset('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/esa_cci_landcover/esa-cci-dominant-landcover-2017.nc')
lc_subset: xr.Dataset = lc.isel(
    latitude=(lc.latitude >= 30) & (lc.latitude <= 90),
    )
value = lc_subset['layer'].values #value meanings refer to google docs
coor_df['lc'] = np.flip(value, axis=0).flatten().astype(int)


# export file
coor_df.to_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/cell_id_table/cell_id_table.csv', encoding='utf-8', index=False)
