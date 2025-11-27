import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cartopy.mpl.ticker as cticker
import cartopy
import cartopy.crs as ccrs
import xarray as xr
from matplotlib.colors import ListedColormap
import geopandas

import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info
from matplotlib.lines import Line2D

region_extent = [-170, -100, 50, 75]
subtitle_loc = [-165, 72]
ccrs_plot = ccrs.PlateCarree()

def add_above_boundaries(ax):
    above_domain = geopandas.read_file(
        f"/resnick/groups/carnegie_poc/michalak-lab/nasa-above/data/domain/ABoVE_reference_grid_v2_1527/data/ABoVE_Study_Domain/ABoVE_Study_Domain.shp"
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

    proj = ccrs.AlbersEqualArea( # the plot actually does not use it
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

def setup_plot(ax, region_extent, ccrs_plot, axes=True):
    ax.set_extent(region_extent, ccrs_plot)
    ax.coastlines(linewidth=0.5)
    ax.set_aspect(2)

    ax.set_xticks(np.arange(-160, -80, 20), crs=ccrs_plot)
    ax.set_yticks(np.arange(50, 80, 5), crs=ccrs_plot)
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(axis='both', which='major', labelsize=14, length=8)
    if not axes:
        # ax.set_xticklabels([])
        ax.set_yticklabels([])

    return ax

fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': ccrs_plot})

# Fig 1a: Land cover map
ax = axs[0]
ax = setup_plot(ax, region_extent, ccrs_plot, axes=True)
ax.text(subtitle_loc[0], subtitle_loc[1], '(a)', fontsize=15)

lc = xr.open_dataset('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/esa_cci_landcover/esa-cci-dominant-landcover-2017.nc').layer
lc_reclassify = xr.where(lc > -999, 0, np.nan)
lc_reclassify = xr.where(lc == 5, 1, lc_reclassify)  # ENF
lc_reclassify = xr.where(lc == 7, 2, lc_reclassify)  # shrub
lc_reclassify = xr.where((lc >= 8) & (lc <= 10), 3, lc_reclassify)  # grass

ABoVE_mask = xr.open_dataset('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/above_mask/above_ext.nc')
ABoVE_mask = ABoVE_mask.rename({'lat': 'latitude', 'lon': 'longitude'})
lc_reclassify = lc_reclassify.where(ABoVE_mask['ids'] == 0)

lons = lc_reclassify["longitude"].values
lats = lc_reclassify["latitude"].values
lon_grid, lat_grid = np.meshgrid(lons, lats)

cmap = ListedColormap(["lightgrey", "#05450a", "#c6b044", "#ffafdc"])
cp = ax.pcolormesh(lon_grid, lat_grid, lc_reclassify, cmap=cmap)

legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#05450a', markersize=10, label='Forests'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#c6b044', markersize=10, label='Shrubs'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffafdc', markersize=10, label='Tundra'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgrey', markersize=10, label='Others')
]
ax.legend(handles=legend_elements, loc='lower center', fontsize=14, ncol=2, columnspacing=0.5, bbox_to_anchor=(0.5, -0.6))

add_above_boundaries(ax)

# Fig 1b: Flight tracks of each year
ax = axs[1]
ax = setup_plot(ax, region_extent, ccrs_plot, axes=False)
ax.text(subtitle_loc[0], subtitle_loc[1], '(b)', fontsize=15)

for year, color in zip([2012, 2013, 2014, 2017], ['blue', 'orange', 'green', 'purple']):
    start_month, end_month, campaign_name = get_campaign_info(year)
    df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
    df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

    local_hour = pd.to_datetime(df_airborne['footprint_time_AKT'], utc=True).dt.tz_convert('America/Anchorage').dt.hour

    mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
                       (df_influence['ABoVE_influence_fraction'] > 0.5) &
                       (df_influence['ocean_influence_fraction'] < 0.3) &
                       (df_airborne['CO2_change'] < 30) &
                       (df_airborne['CO_change'] < 40))[0].tolist()

    df_year = pd.concat((df_airborne, df_influence), axis=1)
    df_year = df_year.loc[mask_id]

    ax.scatter(df_year['airborne_lon'], df_year['airborne_lat'], color=color, s=0.2, label=year)

add_above_boundaries(ax)
ax.legend(loc='lower center', fontsize=14, ncol=2, columnspacing=0.5, bbox_to_anchor=(0.5, -0.6), markerscale=15)


# Fig 1c: Aggregated footprint sensitivity
ax = axs[2]
ax = setup_plot(ax, region_extent, ccrs_plot, axes=False)
ax.text(subtitle_loc[0], subtitle_loc[1], '(c)', fontsize=15)

# averaged influence from all years
filestr = '_selected'
influence_all_year = xr.open_dataset(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/arctic_cap_airborne/h_matrix/summarized_footprint_sensitivity/influence_mean_allyears{filestr}.nc').influence

lons = influence_all_year["longitude"].values
lats = influence_all_year["latitude"].values
lon_grid, lat_grid = np.meshgrid(lons, lats)
cp = ax.pcolormesh(lon_grid, lat_grid, np.log10(influence_all_year), vmin=-4, vmax=-2, cmap='Purples')

# cax = fig.add_axes([0.7, 0.01, 0.2, 0.03])
cax = fig.add_axes([0.69, 0.05, 0.2, 0.03])

cb = fig.colorbar(cp, cax=cax, orientation="horizontal")
cb.ax.tick_params(labelsize=14)

# cb.ax.text(0.5, 2.8, "(Normalized unit)", fontsize=11, ha='center', va='top', transform=cb.ax.transAxes)
# cb.set_label("log (footprint sensitivity)", fontsize=15)
cb.set_label(r"ppm $(\mu mol\ m^{-2}\ s^{-1})^{-1}$", fontsize=15, labelpad=10)

# cb.ax.xaxis.set_label_coords(0.5, 4.5)
cb.ax.xaxis.set_label_coords(0.5, 3.5)

# cb.set_ticks([0, 1, 2])
# cb.set_ticklabels(['$10^0$', '$10^1$', '$10^2$'])
# cb.set_ticklabels([0, 0.5, 1])
cb.set_ticks([-4, -3, -2])
cb.set_ticklabels([ r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'])

add_above_boundaries(ax)

plt.subplots_adjust(wspace=0.2)

fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig1.png', dpi=300, bbox_inches='tight')
fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig1.pdf', dpi=300, bbox_inches='tight')
plt.show()