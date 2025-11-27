# plot soil temperature output from selected TRENDY models
# modified from plot_spatial.py

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/src')
from functions import get_campaign_info, read_TRENDYv11, read_TRENDYv9, read_inversions, read_fossil_fire


# selected models with tsl output
model_names = ['CLASSIC', 'ISBA-CTRIP',  'JSBACH', 'JULES','CABLE-POP'] #, 'ISAM'

# regional mask
ABoVE_mask = xr.open_dataset('/central/groups/carnegie_poc/jwen2/ABoVE/above_mask/above_ext.nc')
ABoVE_mask_subset = ABoVE_mask.isel(
                    lat=(ABoVE_mask.lat >= 30) & (ABoVE_mask.lat <= 90),
                    )
mask_array = ABoVE_mask_subset['ids'].values #360 lat x 720 lon

for var_name in ['tsl', 'NEE', 'GPP']:
    if var_name == 'tsl':
        zlim1 = -20
        zlim2 = 20
        cmap = 'BrBG_r'
    elif var_name == 'NEE':
        zlim1 = -3
        zlim2 = 3
        cmap = 'BrBG_r'
    elif var_name == 'GPP':
        zlim1 = 0
        zlim2 = 10
        cmap = 'viridis'

    for year in [2017]: #2012, 2013, 2014, 2017

        start_month, end_month, campaign_name = get_campaign_info(year)
        # create dir
        dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/spatial/"
        if not os.path.exists(dir0):
            os.makedirs(dir0)
            
        for month in np.arange(4,12):

            fig, ax = plt.subplots(nrows=3,ncols=2,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,8))


            ax = ax.flatten()
            subplot_id = -1
            for model_name in model_names:
                subplot_id += 1

                # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
                # same order as in the cell_id_table.csv

                if var_name == 'tsl':
                    ds_subset = read_TRENDYv11(model_name, 'tsl', year, month) - 273.15 # convert the unit from K to C
                elif var_name == 'NEE':
                    ds_subset_gpp = read_TRENDYv11(model_name, 'gpp', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset_ra = read_TRENDYv11(model_name, 'ra', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset_rh = read_TRENDYv11(model_name, 'rh', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset = ds_subset_ra + ds_subset_rh - ds_subset_gpp
                elif var_name == 'GPP':
                    ds_subset = read_TRENDYv11(model_name, 'gpp', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1

                #ABoVE mask
                ds_subset = ds_subset.where(mask_array == 0.0) # only ABoVE core region

                # plot
                cm = plt.get_cmap(cmap)
                cm.set_bad("lightgrey")
                ds_subset.plot(ax=ax[subplot_id], transform=ccrs.PlateCarree(),
                        cbar_kwargs={'shrink': 0.4}, vmin=zlim1, vmax=zlim2, cmap=cm)
                ax[subplot_id].set_extent([-170, -100, 50, 75], ccrs.PlateCarree())
                gl = ax[subplot_id].gridlines(draw_labels=True) #linewidth=2, color='gray', alpha=0.5, linestyle='--'
                gl.top_labels = False
                gl.right_labels = False
                ax[subplot_id].coastlines()
                ax[subplot_id].set_aspect(2)
                ax[subplot_id].set_title(model_name)
            
            # remove empty panels
            if subplot_id<len(ax):
                for ax1 in ax.flat[(subplot_id+1):]:
                    ax1.remove()
            plt.savefig(f'{dir0}/TRENDYv11_selected_{var_name}_{year}_{month}.png', dpi=100, bbox_inches='tight')
            plt.show()


            # plot a histgram
            fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(8,8))


            ax = ax.flatten()
            subplot_id = -1
            for model_name in model_names:
                subplot_id += 1

                # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
                # same order as in the cell_id_table.csv

                if var_name == 'tsl':
                    ds_subset = read_TRENDYv11(model_name, 'tsl', year, month) - 273.15 # convert the unit from K to C
                elif var_name == 'NEE':
                    ds_subset_gpp = read_TRENDYv11(model_name, 'gpp', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset_ra = read_TRENDYv11(model_name, 'ra', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset_rh = read_TRENDYv11(model_name, 'rh', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset = ds_subset_ra + ds_subset_rh - ds_subset_gpp
                elif var_name == 'GPP':
                    ds_subset = read_TRENDYv11(model_name, 'gpp', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1

                #ABoVE mask
                ds_subset = ds_subset.where(mask_array == 0.0) # only ABoVE core region

                # plot
                ax[subplot_id].hist(ds_subset.values.flatten(), bins=50, range=(zlim1, zlim2))
                ax[subplot_id].set_title(model_name)
            
            # remove empty panels
            if subplot_id<len(ax):
                for ax1 in ax.flat[(subplot_id+1):]:
                    ax1.remove()
            plt.savefig(f'{dir0}/TRENDYv11_selected_{var_name}_{year}_{month}_hist.png', dpi=100, bbox_inches='tight')
            plt.show()
