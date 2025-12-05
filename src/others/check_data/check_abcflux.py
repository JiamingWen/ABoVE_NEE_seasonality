import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import rioxarray
import pyproj
from pyproj import CRS

f = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/abcflux_upscaled/CO2Fluxes_Arctic_Boreal_Domain/data/CO2Fluxes_Arctic_Boreal_GPP_2017.nc"

ds = xr.open_dataset(f)
crs_variable = ds.get("crs")
crs_wkt = crs_variable.attrs.get("crs_wkt", None)
crs = CRS.from_wkt(crs_wkt)
ds = ds.rio.write_crs(crs, inplace=True)
ds = ds.rename({"easting": "x", "northing": "y"})
ds2 = ds['NEE']
ds2_proj = ds2.rio.reproject("EPSG:4326")

ds2_proj_2 = ds2_proj
ds2_proj_2 = ds2_proj_2.rename({"x": "lon", "y": "lat"})
output_file = "/resnick/groups/carnegie_poc/jwen2/ABoVE/test.nc"
ds2_proj_2.attrs.pop('grid_mapping', None)
ds2_proj_2.to_netcdf(output_file)

fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([-170, -100, 50, 75], ccrs.PlateCarree())
# gl = ax.gridlines(draw_labels=True) #linewidth=2, color='gray', alpha=0.5, linestyle='--'
# gl.xlabels_top = False
# gl.ylabels_right = False
# ax.coastlines()
# cm = plt.get_cmap("viridis")
# cm.set_bad("lightgrey")
ds2_proj_2.plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_aspect(2)
plt.show()
plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/test.png', dpi=100, bbox_inches='tight')



output_file = "/resnick/groups/carnegie_poc/jwen2/ABoVE/test.nc"
ds_new = xr.open_dataset(output_file)

fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([-170, -100, 50, 75], ccrs.PlateCarree())
# gl = ax.gridlines(draw_labels=True) #linewidth=2, color='gray', alpha=0.5, linestyle='--'
# gl.xlabels_top = False
# gl.ylabels_right = False
# ax.coastlines()
# cm = plt.get_cmap("viridis")
# cm.set_bad("lightgrey")
ds_new['NEE'].plot(ax=ax, transform=ccrs.PlateCarree())
# ax.set_aspect(2)
plt.show()
plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/test.png', dpi=100, bbox_inches='tight')



ds_regrid = xr.open_dataset('/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/abcflux_upscaled/half-degree/CO2Fluxes_Arctic_Boreal_NEE_2017-half-degree-ABoVE.nc')
ds_regrid = ds_regrid.where((ds_regrid.time.dt.month == 7), drop=True)
fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-170, -100, 50, 75], ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True) #linewidth=2, color='gray', alpha=0.5, linestyle='--'
gl.xlabels_top = False
gl.ylabels_right = False
ax.coastlines()
cm = plt.get_cmap("viridis")
cm.set_bad("lightgrey")
ds_regrid['NEE'].plot(ax=ax, transform=ccrs.PlateCarree())
ax.set_aspect(2)
plt.show()


# check projected data
# ds_proj = xr.open_dataset('/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/abcflux_upscaled/half-degree/projected_NEE_2001.nc')
ds_proj = xr.open_dataset('/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/abcflux_upscaled/half-degree/projected.nc')
ds_proj = ds_proj.where((ds_proj.time.dt.month == 7), drop=True)
fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-170, -100, 50, 75], ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True) #linewidth=2, color='gray', alpha=0.5, linestyle='--'
gl.xlabels_top = False
gl.ylabels_right = False
ax.coastlines()
cm = plt.get_cmap("viridis")
cm.set_bad("lightgrey")
ds_proj['NEE'].plot(ax=ax, transform=ccrs.PlateCarree())
ax.set_aspect(2)
plt.show()


# check regridded data
# ds_regrid = xr.open_dataset('/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/abcflux_upscaled/half-degree/old2/CO2Fluxes_Arctic_Boreal_NEE_2017-half-degree.nc')
ds_regrid = xr.open_dataset('/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/abcflux_upscaled/half-degree/CO2Fluxes_Arctic_Boreal_GPP_2001-half-degree.nc')
ds_regrid = ds_regrid.where((ds_regrid.time.dt.month == 7), drop=True)
fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([-170, -100, 50, 75], ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True) #linewidth=2, color='gray', alpha=0.5, linestyle='--'
gl.xlabels_top = False
gl.ylabels_right = False
ax.coastlines()
cm = plt.get_cmap("viridis")
cm.set_bad("lightgrey")
ds_regrid['GPP'].plot(ax=ax, transform=ccrs.PlateCarree())
ax.set_aspect(2)
plt.show()