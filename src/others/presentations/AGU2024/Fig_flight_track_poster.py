import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs
import xarray as xr
from matplotlib.colors import ListedColormap
import geopandas

import os
os.chdir('/resnick/groups/carnegie_poc/jwen2/ABoVE/src')
from functions import get_campaign_info

region_extent = [-170, -100, 50, 75]; subtitle_loc = [-165, 72] # core
ccrs_plot = ccrs.PlateCarree()


'''a function to draw boundary of ABoVE core domain'''
def add_above_boundaries(ax):

    # ABoVE study domain
    above_domain = geopandas.read_file(
        f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/domain/ABoVE_reference_grid_v2_1527/data/ABoVE_Study_Domain/ABoVE_Study_Domain.shp"
    )
    above_core_geometry = above_domain.loc[
        above_domain["Region"] == "Core Region", "geometry"
    ]

    globe = ccrs.Globe(
        datum="NAD83",
        ellipse="GRS80",
        semimajor_axis=6378137.0,
        semiminor_axis=6356752.314140356,
        inverse_flattening=298.257222101,
    )

    proj = ccrs.AlbersEqualArea(
    central_longitude=-96.0,
    central_latitude=40,
    standard_parallels=(50.0, 70.0),
    globe=globe,
    )

    ax.add_geometries(
        above_core_geometry,
        crs=proj,
        edgecolor="red",
        facecolor="none",
        zorder=1,
        linewidth=1,
        linestyle="solid"
    )


'''a function to define the plot settings'''
def setup_plot(ax, region_extent, ccrs_plot):
    ax.set_extent(region_extent, ccrs_plot)
    gl = ax.gridlines(draw_labels=True)
    ax.coastlines(linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 14}
    gl.ylabel_style = {'fontsize': 14}
    ax.set_aspect(2)
    return ax


# Fig 1c: aggregated footprint sensitivit
fig = plt.figure(figsize=(6,3))
ax = plt.axes(projection=ccrs_plot)
ax = setup_plot(ax, region_extent, ccrs_plot)
plt.text(subtitle_loc[0], subtitle_loc[1], '(c)', fontsize=15)

filestr = '_selected'
for year in [2012,2013,2014,2017]:

    start_month, end_month, campaign_name = get_campaign_info(year)
    influence = xr.open_dataset(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/h_matrix/influence_sum{year}{filestr}.nc').influence

    if year == 2012:
        influence_sum = influence
    else:
        influence_sum = influence_sum + influence

lons = influence_sum["longitude"].values
lats = influence_sum["latitude"].values
lon_grid, lat_grid = np.meshgrid(lons, lats)
cax = fig.add_axes([0.25, -0.035, 0.5, 0.05])  
cp = ax.pcolormesh(lon_grid, lat_grid, np.log10(influence_sum), vmin=0, vmax=2, cmap='Purples') #OrRd
cb = fig.colorbar(cp, cax=cax, orientation="horizontal")
cb.ax.tick_params(labelsize=14)
cb.set_label(
    "Footprint sensitivity (arbitrary unit)",
    fontsize=12,
)
cb.set_ticks([0, 1, 2])
cb.set_ticklabels(['$10^0$', '$10^1$', '$10^2$'])

# add ABoVE boundary
add_above_boundaries(ax)

plt.show()
fig.savefig('/resnick/groups/carnegie_poc/jwen2/ABoVE/result/figures/aggregated_sensitivity_poster.png', dpi=300, bbox_inches='tight') # Save the figure


