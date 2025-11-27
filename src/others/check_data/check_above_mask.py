import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

# regional mask
ABoVE_mask_original = xr.open_dataset('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/above_mask/above_ext.nc')
ABoVE_mask_original = ABoVE_mask_original['ids']
ABoVE_mask = ABoVE_mask_original
ABoVE_mask = xr.where(ABoVE_mask_original == 255, 2, ABoVE_mask_original)

# plot
fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax[subplot_id].set_extent([-170, -100, 50, 75], ccrs.PlateCarree())
ax.set_extent([-170, -80, 50, 80], ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=True) #linewidth=2, color='gray', alpha=0.5, linestyle='--'
# ax.set_global()
ax.coastlines()
gl.top_labels = False
gl.right_labels = False
ax.set_aspect(2)
ABoVE_mask.plot(ax=ax, transform=ccrs.PlateCarree(),
         cbar_kwargs={'shrink': 0.4})
plt.show()
plt.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/above_mask/plot_above_mask.png', dpi=100, bbox_inches='tight')
