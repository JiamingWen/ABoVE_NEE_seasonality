'''extract seasonal cycle of different datasets'''

import numpy as np
import pandas as pd
import xarray as xr
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info, read_TRENDYv11, read_TRENDYv9, read_inversions, read_inversions_prior, read_remote_sensing, read_gosif_gpp, read_fluxcom_x, read_abcflux, read_fossil_fire, read_MODIS_VI, read_GOME2_SIF


# year = 2012 # 2012 2013 2014 2017

for year in [2012, 2013, 2014, 2017]:

    start_month, end_month, campaign_name = get_campaign_info(year)

    # create dir
    dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/"
    if not os.path.exists(dir0):
        os.makedirs(dir0)

    # some anciliary datasets
    # regional mask and land cover table
    cell_id_table = pd.read_csv('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/cell_id_table/cell_id_table.csv')

    # footprint sensitivity map
    influence = xr.open_dataset(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/summarized_footprint_sensitivity/influence_sum{year}_selected.nc')

    for regionname in ['ABoVEcore']: #, 'ABoVEcoreextended'

        if regionname == 'ABoVEcore':
            region_mask = np.where(cell_id_table['ABoVE'] == 0)[0]
        elif regionname == 'ABoVEcoreextended':
            region_mask = np.where(cell_id_table['ABoVE'] != 255)[0]

        for lcname in ['alllc', 'forest', 'shrub', 'tundra']: #'alllc', 'forest', 'shrub', 'tundra'
            if lcname == 'alllc':
                lc_mask = cell_id_table['cell_id']
            elif lcname == 'forestshrub':
                lc_mask = [i for i, val in enumerate(cell_id_table['lc']) if val in [5,7]]
            elif lcname == 'forest':
                lc_mask = [i for i, val in enumerate(cell_id_table['lc']) if val in [5]]
            elif lcname == 'shrub':
                lc_mask = [i for i, val in enumerate(cell_id_table['lc']) if val in [7]]
            elif lcname == 'tundra':
                lc_mask = [i for i, val in enumerate(cell_id_table['lc']) if val in [8,9,10]]

            for weightname in ['unweighted']: #'unweighted', 'weighted'
                if weightname == 'unweighted':
                    weight = np.ones(120*720)
                elif weightname == 'weighted':
                    weight = influence.influence.values.flatten()
                
                for model_type in ['TRENDYv11', 'TRENDYv9', 'inversions', 'reference', 'TRENDYv11GPP', 'TRENDYv11Ra', 'TRENDYv11Rh', 'TRENDYv11LAI', 'GPPobservations', 'NEEobservations', 'Recoobservations', 'Fires', 'inversionsNEE', 'TRENDYv11tsl', 'inversions-prior', 'inversionsNEE-prior']: #'TRENDYv11', 'TRENDYv9', 'inversions', 'reference', 'TRENDYv11GPP', 'TRENDYv11Ra', 'TRENDYv11Rh', 'TRENDYv11LAI', 'GPPobservations', 'NEEobservations', 'Recoobservations', 'Fires', 'inversionsNEE', 'TRENDYv11tsl', 'inversions-prior', 'inversionsNEE-prior'
                    print(year, regionname, lcname, weightname, model_type)
                    if model_type in ['TRENDYv11', 'TRENDYv11GPP', 'TRENDYv11Ra', 'TRENDYv11Rh', 'TRENDYv11LAI']:
                        model_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
                    elif model_type == 'TRENDYv9':
                        model_names = ['CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT']
                    elif model_type in ['inversions', 'inversionsNEE', 'inversions-prior', 'inversionsNEE-prior']:
                        model_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
                    elif model_type in ['reference']:
                        model_names = ['APAR', 'PAR', 'FPAR', 'LAI', 'NDVI', 'EVI', 'GOME2_SIF']
                    elif model_type == 'GPPobservations':
                        model_names = ['GOSIF-GPP', 'FluxCOM-X-GPP', 'ABCflux-GPP']
                    elif model_type == 'NEEobservations':
                        model_names = ['FluxCOM-X-NEE', 'ABCflux-NEE']
                    elif model_type == 'Recoobservations':
                        model_names = ['ABCflux-Reco']
                    elif model_type == 'Fires':
                        model_names = ['Fires']
                    elif model_type == 'TRENDYv11tsl':
                        model_names = ['CLASSIC', 'ISBA-CTRIP',  'JSBACH', 'JULES','CABLE-POP']  #, 'ISAM'

                    seasonal_df = pd.DataFrame()
                    for model_name in model_names:
                        
                        seasonal_vec = []
                        for month in np.arange(1, 13):

                            if model_type == 'TRENDYv11':
                                # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
                                # same order as in the cell_id_table.csv
                                gpp_vec = read_TRENDYv11(model_name, 'gpp', year, month).values.flatten()  #unit: kgC m-2 s-1
                                ra_vec = read_TRENDYv11(model_name, 'ra', year, month).values.flatten()
                                rh_vec = read_TRENDYv11(model_name, 'rh', year, month).values.flatten()
                                nee_vec = ra_vec + rh_vec - gpp_vec
                                nee = nee_vec*1000/12*1e6 #convert unit to μmol m-2 s-1
                                variable = np.nan_to_num(nee, nan=0)

                            elif model_type == 'TRENDYv9':
                                gpp_vec = read_TRENDYv9(model_name, 'gpp', year, month).values.flatten()  #unit: kgC m-2 s-1
                                ra_vec = read_TRENDYv9(model_name, 'ra', year, month).values.flatten()
                                rh_vec = read_TRENDYv9(model_name, 'rh', year, month).values.flatten()
                                nee_vec = ra_vec + rh_vec - gpp_vec
                                nee = nee_vec*1000/12*1e6 #convert unit to μmol m-2 s-1
                                variable = np.nan_to_num(nee, nan=0)

                            elif model_type in ['inversions', 'inversions-prior']:
                                # nbe
                                if model_type == 'inversions':
                                    nbe_vec = read_inversions(model_name, 'land_flux_only_fossil_cement_adjusted', year, month).values.flatten() #unit: PgC/m2/yr
                                else:
                                    nbe_vec = read_inversions_prior(model_name, 'prior_flux_land', year, month).values.flatten()
                                
                                nbe = nbe_vec*1e15/12*1e6/365/24/3600 #convert unit to μmol m-2 s-1
                                
                                if np.isnan(nbe).all():
                                    continue
                                else:
                                    variable = np.nan_to_num(nbe, nan=0)

                            elif model_type == 'TRENDYv11GPP':
                                # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
                                # same order as in the cell_id_table.csv
                                vec = read_TRENDYv11(model_name, 'gpp', year, month).values.flatten()  #unit: kgC m-2 s-1
                                variable = vec*1000/12*1e6 #convert unit to μmol m-2 s-1
                                variable = np.nan_to_num(variable, nan=0)
                            
                            elif model_type == 'TRENDYv11Ra':
                                # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
                                # same order as in the cell_id_table.csv
                                vec = read_TRENDYv11(model_name, 'ra', year, month).values.flatten()  #unit: kgC m-2 s-1
                                variable = vec*1000/12*1e6 #convert unit to μmol m-2 s-1
                                variable = np.nan_to_num(variable, nan=0)
                            
                            elif model_type == 'TRENDYv11Rh':
                                # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
                                # same order as in the cell_id_table.csv
                                vec = read_TRENDYv11(model_name, 'rh', year, month).values.flatten()  #unit: kgC m-2 s-1
                                variable = vec*1000/12*1e6 #convert unit to μmol m-2 s-1
                                variable = np.nan_to_num(variable, nan=0)
                                
                            elif model_type == 'TRENDYv11LAI':
                                # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
                                # same order as in the cell_id_table.csv
                                if model_name == 'LPJ': # no lai output
                                    variable = np.empty((86400))
                                    variable[:] = np.nan
                                else:
                                    variable = read_TRENDYv11(model_name, 'lai', year, month).values.flatten()
                                    variable = np.nan_to_num(variable, nan=0)

                            elif model_type == 'reference':
                                if model_name == 'PAR':
                                    variable = read_remote_sensing('par', 'PAR', year, month).values.flatten()
                                elif model_name == 'FPAR':
                                    variable = read_remote_sensing('fpar', 'Fpar', year, month).values.flatten()
                                    if len(variable) == 0:
                                        variable = np.empty((86400))
                                        variable[:] = np.nan
                                elif model_name == 'LAI':
                                    variable = read_remote_sensing('lai', 'Lai', year, month).values.flatten()
                                    if len(variable) == 0:
                                        variable = np.empty((86400))
                                        variable[:] = np.nan
                                elif model_name == 'APAR':
                                    par_vec = read_remote_sensing('par', 'PAR', year, month).values.flatten()
                                    fpar_vec = read_remote_sensing('fpar', 'Fpar', year, month).values.flatten()
                                    if len(fpar_vec) == 0:
                                        fpar_vec = np.empty((par_vec.shape))
                                        fpar_vec[:] = np.nan
                                    variable = par_vec * fpar_vec
                                elif model_name == 'NDVI':
                                    variable = read_MODIS_VI('NDVI', year, month).values.flatten()
                                    variable = np.nan_to_num(variable, nan=0)
                                elif model_name == 'EVI':
                                    variable = read_MODIS_VI('EVI', year, month).values.flatten()
                                    variable = np.nan_to_num(variable, nan=0)
                                elif model_name == 'GOME2_SIF':
                                    variable = read_GOME2_SIF('dcSIF', year, month).values.flatten()
                                    variable = np.nan_to_num(variable, nan=0)

                            elif model_type == 'GPPobservations':
                                if model_name == 'GOSIF-GPP':
                                    variable = read_gosif_gpp(year, month).values.flatten() # unit: g C m-2 mo-1
                                    variable = variable/30/24/3600/12*1e6 # convert unit to μmol m-2 s-1
                                    variable = np.nan_to_num(variable, nan=0)
                                elif model_name == 'FluxCOM-X-GPP':
                                    variable = read_fluxcom_x('GPP', year, month).values.flatten() # unit gC m-2 d-1
                                    variable = variable/24/3600/12*1e6 # convert unit to μmol m-2 s-1
                                    variable = np.nan_to_num(variable, nan=0)
                                elif model_name == 'ABCflux-GPP':
                                    variable = read_abcflux('GPP', year, month).values.flatten() # unit gC m-2 mo-1
                                    variable = variable/30/24/3600/12*1e6 # convert unit to μmol m-2 s-1

                            elif model_type == 'NEEobservations':
                                if model_name == 'FluxCOM-X-NEE':
                                    variable = read_fluxcom_x('NEE', year, month).values.flatten() # unit gC m-2 d-1
                                    variable = variable/24/3600/12*1e6 # convert unit to μmol m-2 s-1
                                    variable = np.nan_to_num(variable, nan=0)
                                elif model_name == 'ABCflux-NEE':
                                    variable = read_abcflux('NEE', year, month).values.flatten() # unit gC m-2 mo-1
                                    variable = variable/30/24/3600/12*1e6 # convert unit to μmol m-2 s-1

                            elif model_type == 'Recoobservations':
                                if model_name == 'ABCflux-Reco':
                                    variable = read_abcflux('Reco', year, month).values.flatten() # unit gC m-2 mo-1
                                    variable = variable/30/24/3600/12*1e6 # convert unit to μmol m-2 s-1
                                
                            elif model_type == 'Fires':
                                variable = read_fossil_fire('fire', 'CO2_emission', year, month).values.flatten() #unit: gCO2 m-2 month-1
                                variable = variable/30/24/3600/44*1e6 #convert unit to μmol m-2 s-1
                            
                            elif model_type in ['inversionsNEE', 'inversionsNEE-prior']:
                                # nbe
                                if model_type == 'inversionsNEE':
                                    nbe_vec = read_inversions(model_name, 'land_flux_only_fossil_cement_adjusted', year, month).values.flatten() #unit: PgC/m2/yr
                                else:
                                    nbe_vec = read_inversions_prior(model_name, 'prior_flux_land', year, month).values.flatten()
                                
                                nbe = nbe_vec*1e15/12*1e6/365/24/3600 #convert unit to μmol m-2 s-1
                                
                                if np.isnan(nbe).all():
                                    continue
                                else:
                                    variable = np.nan_to_num(nbe, nan=0)

                                # fire
                                fire = read_fossil_fire('fire', 'CO2_emission', year, month).values.flatten() #unit: gCO2 m-2 month-1
                                fire = fire/30/24/3600/44*1e6 #convert unit to μmol m-2 s-1

                                variable = nbe - fire

                            elif model_type == 'TRENDYv11tsl':
                                variable = read_TRENDYv11(model_name, 'tsl', year, month).values.flatten()
                                if model_name == 'CABLE-POP':
                                    variable[variable<-1000] = np.nan


                            mask_id0 = [value for value in region_mask if value in lc_mask]
                            mask_id = [i for i in mask_id0 if ~np.isnan(variable[i])] # exclude nan for tsl average
                            variable_subset = variable[mask_id]
                            weight_subset = weight[mask_id]
                            regional_mean = np.sum(variable_subset * weight_subset) / np.sum(weight_subset)
                            seasonal_vec.append(regional_mean)

                        seasonal_df = pd.concat((seasonal_df, pd.DataFrame({model_name: seasonal_vec})), axis=1)

                    seasonal_df.to_csv(f'{dir0}/seasonal_{year}_{model_type}_{regionname}_{lcname}_{weightname}.csv', encoding='utf-8', index=False)
