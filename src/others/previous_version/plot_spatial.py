# plot spatial maps of NEE from TRENDY or linear regression models

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/src')
from functions import get_campaign_info, read_TRENDYv11, read_TRENDYv9, read_inversions, read_fossil_fire


# regional mask
ABoVE_mask = xr.open_dataset('/central/groups/carnegie_poc/jwen2/ABoVE/above_mask/above_ext.nc')
ABoVE_mask_subset = ABoVE_mask.isel(
                    lat=(ABoVE_mask.lat >= 30) & (ABoVE_mask.lat <= 90),
                    )
mask_array = ABoVE_mask_subset['ids'].values #120 lat x 720 lon

zlim1 = -3
zlim2 = 3
cmap = 'BrBG_r'

for year in [2012, 2013, 2014, 2017]: #

    start_month, end_month, campaign_name = get_campaign_info(year)
    # create dir
    dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/spatial/"
    if not os.path.exists(dir0):
        os.makedirs(dir0)

    for model_type in ['TRENDYv11','inversionsNEE']: #'TRENDYv11', 'inversionsNEE', 'TRENDYv9', 'inversions'
        
        for month in np.arange(4,12):

            if model_type == 'TRENDYv11':
                model_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
                fig, ax = plt.subplots(nrows=4,ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(22,15))
            elif model_type == 'TRENDYv9':
                model_names = ['CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT']
                fig, ax = plt.subplots(nrows=3,ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(22,11))
            elif model_type in ['inversions', 'inversionsNEE']:
                model_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
                fig, ax = plt.subplots(nrows=4,ncols=4,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(22,15))

            ax = ax.flatten()
            subplot_id = -1
            for model_name in model_names:
                subplot_id += 1

                if model_type == 'TRENDYv11':
                    # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
                    # same order as in the cell_id_table.csv
                    ds_subset_gpp = read_TRENDYv11(model_name, 'gpp', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset_ra = read_TRENDYv11(model_name, 'ra', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset_rh = read_TRENDYv11(model_name, 'rh', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset = ds_subset_ra + ds_subset_rh - ds_subset_gpp

                elif model_type == 'TRENDYv9':
                    ds_subset_gpp = read_TRENDYv9(model_name, 'gpp', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset_ra = read_TRENDYv9(model_name, 'ra', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset_rh = read_TRENDYv9(model_name, 'rh', year, month) *1000/12*1e6 #convert unit to μmol m-2 s-1
                    ds_subset = ds_subset_ra + ds_subset_rh - ds_subset_gpp

                elif model_type == 'inversions':
                    ds_subset = read_inversions(model_name, 'land_flux_only_fossil_cement_adjusted', year, month)*1e15/12*1e6/365/24/3600 #convert unit to μmol m-2 s-1
                
                elif model_type == 'inversionsNEE':
                    nbe = read_inversions(model_name, 'land_flux_only_fossil_cement_adjusted', year, month)*1e15/12*1e6/365/24/3600 #convert unit to μmol m-2 s-1
                    fire = read_fossil_fire('fire', 'CO2_emission', year, month)/30/24/3600/44*1e6 #convert unit to μmol m-2 s-1
                    ds_subset = nbe - fire

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
            plt.savefig(f'{dir0}/{model_type}_{year}_{month}.png', dpi=100, bbox_inches='tight')
            plt.show()