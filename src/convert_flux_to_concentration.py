'''convert NEE surface flux or remote sensing fields to the concentration space'''

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info, read_H_matrix, subset_30N_90N, read_fossil, read_fire, read_TRENDYv11, read_TRENDYv9, read_inversions, read_remote_sensing, read_x_base_monthly, read_abcflux, read_gosif_gpp, read_MODIS_VI, read_GOME2_SIF, read_inversions_prior, read_ocean_fluxes

year = 2017 # 2012 2013 2014 2017

start_month, end_month, campaign_name = get_campaign_info(year)

dir_out = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# read observations
receptor_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
n_receptor = receptor_df.shape[0]


# mask for land pixels
cell_id_table = pd.read_csv('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/cell_id_table/cell_id_table.csv')
land_cellnum_list = np.where(cell_id_table['land']==1)[0]


# read H matrix
h_matrix = read_H_matrix(year, n_receptor, land_cellnum_list)


# fossil fuel emissions
result_df = pd.DataFrame()
for data_name in ['odiac2022', 'gridfed2024']:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)

        fossil_vec = subset_30N_90N(read_fossil(data_name, year, month)).values.flatten()[land_cellnum_list] 
        
        if data_name == 'odiac2022':
            variable = fossil_vec/24/3600/12*1e6 #convert unit from gC/m2/d to μmol m-2 s-1
        elif data_name == 'gridfed2024':
            variable = fossil_vec*1000/30/24/3600/44*1e6 #convert unit from kg CO2 m-2 month-1 to μmol m-2 s-1

        if month == start_month:
            X_matrix = variable
        else:
            X_matrix = np.concatenate((X_matrix, variable), axis=0)

    CO2_change = h_matrix @ X_matrix
    result_df[f'{data_name}'] = CO2_change
result_df.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_fossil.csv', encoding='utf-8', index=False)


# fire emissions
result_df = pd.DataFrame()
for data_name in ['gfed4.1', 'gfed5']:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)

        fire_vec = subset_30N_90N(read_fire(data_name, year, month)).values.flatten()[land_cellnum_list] #unit: gCO2 m-2 month-1
        variable = fire_vec/30/24/3600/44*1e6 #convert unit to μmol m-2 s-1

        if month == start_month:
            X_matrix = variable
        else:
            X_matrix = np.concatenate((X_matrix, variable), axis=0)

    CO2_change = h_matrix @ X_matrix
    result_df[f'{data_name}'] = CO2_change
result_df.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_fire.csv', encoding='utf-8', index=False)


# TRENDY v11
TRENDYv11_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
result_df_NEE = pd.DataFrame()
result_df_GPP = pd.DataFrame()
result_df_NPP = pd.DataFrame()
result_df_Reco = pd.DataFrame()

for data_name in TRENDYv11_names:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)

        # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
        # same order as in the cell_id_table.csv
        GPP = subset_30N_90N(read_TRENDYv11(data_name, 'gpp', year, month)).values.flatten()[land_cellnum_list] *1000/12*1e6 #convert unit from kg m-2 s-1 to μmol m-2 s-1
        Ra = subset_30N_90N(read_TRENDYv11(data_name, 'ra', year, month)).values.flatten()[land_cellnum_list] *1000/12*1e6 #convert unit from kg m-2 s-1 to μmol m-2 s-1
        Rh = subset_30N_90N(read_TRENDYv11(data_name, 'rh', year, month)).values.flatten()[land_cellnum_list] *1000/12*1e6 #convert unit from kg m-2 s-1 to μmol m-2 s-1
        
        GPP = np.nan_to_num(GPP, nan=0)
        Ra = np.nan_to_num(Ra, nan=0)
        Rh = np.nan_to_num(Rh, nan=0)

        NEE = Ra + Rh - GPP
        NPP = GPP - Ra
        Reco = Ra + Rh

        if month == start_month:
            X_matrix_NEE = NEE
            X_matrix_GPP = GPP
            X_matrix_NPP = NPP
            X_matrix_Reco = Reco
        else:
            X_matrix_NEE = np.concatenate((X_matrix_NEE, NEE), axis=0)
            X_matrix_GPP = np.concatenate((X_matrix_GPP, GPP), axis=0)
            X_matrix_NPP = np.concatenate((X_matrix_NPP, NPP), axis=0)
            X_matrix_Reco = np.concatenate((X_matrix_Reco, Reco), axis=0)


    CO2_change_NEE = h_matrix @ X_matrix_NEE
    CO2_change_GPP = h_matrix @ X_matrix_GPP
    CO2_change_NPP = h_matrix @ X_matrix_NPP
    CO2_change_Reco = h_matrix @ X_matrix_Reco
    
    result_df_NEE[f'{data_name}'] = CO2_change_NEE
    result_df_GPP[f'{data_name}'] = CO2_change_GPP
    result_df_NPP[f'{data_name}'] = CO2_change_NPP
    result_df_Reco[f'{data_name}'] = CO2_change_Reco

result_df_NEE.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_TRENDYv11.csv', encoding='utf-8', index=False)
result_df_GPP.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_TRENDYv11GPP.csv', encoding='utf-8', index=False)
result_df_NPP.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_TRENDYv11NPP.csv', encoding='utf-8', index=False)
result_df_Reco.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_TRENDYv11Reco.csv', encoding='utf-8', index=False)


# TRENDY v9
TRENDYv9_names = ['CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT']
result_df = pd.DataFrame()
for data_name in TRENDYv9_names:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)

        # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
        # same order as in the cell_id_table.csv
        gpp_vec = subset_30N_90N(read_TRENDYv9(data_name, 'gpp', year, month)).values.flatten()[land_cellnum_list]  #unit: kgC m-2 s-1
        ra_vec = subset_30N_90N(read_TRENDYv9(data_name, 'ra', year, month)).values.flatten()[land_cellnum_list] 
        rh_vec = subset_30N_90N(read_TRENDYv9(data_name, 'rh', year, month)).values.flatten()[land_cellnum_list] 
        nee_vec = ra_vec + rh_vec - gpp_vec
        nee = nee_vec*1000/12*1e6 #convert unit to μmol m-2 s-1
        NEE = np.nan_to_num(nee, nan=0)

        if month == start_month:
            X_matrix = NEE
        else:
            X_matrix = np.concatenate((X_matrix, NEE), axis=0)

    CO2_change = h_matrix @ X_matrix
    result_df[f'{data_name}'] = CO2_change
result_df.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_TRENDYv9.csv', encoding='utf-8', index=False)


# inversions from GCB2023
inversion_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
result_df = pd.DataFrame()
for data_name in inversion_names:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)
        
        # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
        # same order as in the cell_id_table.csv
        nee_vec = subset_30N_90N(read_inversions(data_name, 'land_flux_only_fossil_cement_adjusted', year, month)).values.flatten()[land_cellnum_list] #unit: PgC/m2/yr
        nee = nee_vec*1e15/12*1e6/365/24/3600 #convert unit to μmol m-2 s-1

        if np.isnan(nee).all():
            variable = nee
        else:
            variable = np.nan_to_num(nee, nan=0)

        if month == start_month:
            X_matrix = variable
        else:
            X_matrix = np.concatenate((X_matrix, variable), axis=0)

    CO2_change = h_matrix @ X_matrix
    result_df[f'{data_name}'] = CO2_change
result_df.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_inversions.csv', encoding='utf-8', index=False)


# remote sensing reference
reference_names = ['APAR', 'PAR', 'FPAR', 'LAI', 'NDVI', 'EVI', 'GOME2_SIF']
result_df = pd.DataFrame()
for data_name in reference_names:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)
        
        # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
        # same order as in the cell_id_table.csv

        if data_name == 'PAR':
            variable = subset_30N_90N(read_remote_sensing('par', 'PAR', year, month)).values.flatten()[land_cellnum_list]
        elif data_name == 'FPAR':
            variable = subset_30N_90N(read_remote_sensing('fpar', 'Fpar', year, month)).values.flatten()[land_cellnum_list]
        elif data_name == 'LAI':
            variable = subset_30N_90N(read_remote_sensing('lai', 'Lai', year, month)).values.flatten()[land_cellnum_list]
        elif data_name == 'APAR':
            par_vec = subset_30N_90N(read_remote_sensing('par', 'PAR', year, month)).values.flatten()[land_cellnum_list]
            fpar_vec = subset_30N_90N(read_remote_sensing('fpar', 'Fpar', year, month)).values.flatten()[land_cellnum_list]
            variable = par_vec * fpar_vec
        elif data_name == 'NDVI':
            variable = subset_30N_90N(read_MODIS_VI('NDVI', year, month)).values.flatten()[land_cellnum_list]
        elif data_name == 'EVI':
            variable = subset_30N_90N(read_MODIS_VI('EVI', year, month)).values.flatten()[land_cellnum_list]
        elif data_name == 'GOME2_SIF':
            variable = subset_30N_90N(read_GOME2_SIF('dcSIF', year, month)).values.flatten()[land_cellnum_list]
        
        variable = np.nan_to_num(variable, nan=0)

        if month == start_month:
            X_matrix = variable
        else:
            X_matrix = np.concatenate((X_matrix, variable), axis=0)

    CO2_change = h_matrix @ X_matrix
    result_df[f'{data_name}'] = CO2_change
result_df.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_reference.csv', encoding='utf-8', index=False)


# upscaled EC datasets - NEE
UpscaledEC_names = ['X-BASE', 'ABCflux']
result_df = pd.DataFrame()
for data_name in UpscaledEC_names:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)
        
        # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
        # same order as in the cell_id_table.csv

        if data_name == 'X-BASE':
            variable = subset_30N_90N(read_x_base_monthly('NEE', year, month)).values.flatten()[land_cellnum_list] # unit gC m-2 d-1
            variable = variable/24/3600/12*1e6 # convert unit to μmol m-2 s-1
            variable = np.nan_to_num(variable, nan=0)
        elif data_name == 'ABCflux':
            variable = subset_30N_90N(read_abcflux('NEE', year, month)).values.flatten()[land_cellnum_list] # unit gC m-2 mo-1
            variable = variable/30/24/3600/12*1e6 # convert unit to μmol m-2 s-1

        if month == start_month:
            X_matrix = variable
        else:
            X_matrix = np.concatenate((X_matrix, variable), axis=0)

    CO2_change = h_matrix @ X_matrix
    result_df[f'{data_name}'] = CO2_change
result_df.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_UpscaledEC.csv', encoding='utf-8', index=False)


# upscaled EC datasets - GPP
UpscaledEC_GPP_names = ['X-BASE', 'ABCflux']
result_df = pd.DataFrame()
for data_name in UpscaledEC_GPP_names:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)
        
        # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
        # same order as in the cell_id_table.csv
        
        if data_name == 'X-BASE':
            variable = subset_30N_90N(read_x_base_monthly('GPP', year, month)).values.flatten()[land_cellnum_list] # unit gC m-2 d-1
            variable = variable/24/3600/12*1e6 # convert unit to μmol m-2 s-1
            variable = np.nan_to_num(variable, nan=0)

        elif data_name == 'ABCflux':
            variable = subset_30N_90N(read_abcflux('GPP', year, month)).values.flatten()[land_cellnum_list] # unit gC m-2 mo-1
            variable = variable/30/24/3600/12*1e6 # convert unit to μmol m-2 s-1

        if month == start_month:
            X_matrix = variable
        else:
            X_matrix = np.concatenate((X_matrix, variable), axis=0)

    CO2_change = h_matrix @ X_matrix
    result_df[f'{data_name}'] = CO2_change
result_df.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_UpscaledEC_GPP.csv', encoding='utf-8', index=False)


# upscaled EC datasets - Reco
UpscaledEC_Reco_names = ['X-BASE', 'ABCflux', 'ABCflux_upscaled']
result_df = pd.DataFrame()
for data_name in UpscaledEC_Reco_names:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)
        
        # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
        # same order as in the cell_id_table.csv
        
        if data_name == 'X-BASE':
            variable_NEE = subset_30N_90N(read_x_base_monthly('NEE', year, month)).values.flatten()[land_cellnum_list] # unit gC m-2 d-1
            variable_GPP = subset_30N_90N(read_x_base_monthly('GPP', year, month)).values.flatten()[land_cellnum_list]
            variable = variable_GPP + variable_NEE
            variable = variable/24/3600/12*1e6 # convert unit to μmol m-2 s-1
            variable = np.nan_to_num(variable, nan=0)

        elif data_name == 'ABCflux':
            variable_NEE = subset_30N_90N(read_abcflux('NEE', year, month)).values.flatten()[land_cellnum_list] # unit gC m-2 mo-1
            variable_GPP = subset_30N_90N(read_abcflux('GPP', year, month)).values.flatten()[land_cellnum_list]
            variable = variable_GPP + variable_NEE
            variable = variable/30/24/3600/12*1e6 # convert unit to μmol m-2 s-1

        elif data_name == 'ABCflux_upscaled':
            variable = subset_30N_90N(read_abcflux('Reco', year, month)).values.flatten()[land_cellnum_list] # unit gC m-2 mo-1
            variable = variable/30/24/3600/12*1e6 # convert unit to μmol m-2 s-1

        if month == start_month:
            X_matrix = variable
        else:
            X_matrix = np.concatenate((X_matrix, variable), axis=0)

    CO2_change = h_matrix @ X_matrix
    result_df[f'{data_name}'] = CO2_change
result_df.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_UpscaledEC_Reco.csv', encoding='utf-8', index=False)


# GPP observations
GPPobservations_names = ['GOSIF-GPP']
result_df = pd.DataFrame()
for data_name in GPPobservations_names:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)
        
        # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
        # same order as in the cell_id_table.csv

        if data_name == 'GOSIF-GPP':
            variable = subset_30N_90N(read_gosif_gpp(year, month)).values.flatten()[land_cellnum_list] # unit: g C m-2 mo-1
            variable = variable/30/24/3600/12*1e6 # convert unit to μmol m-2 s-1
            variable = np.nan_to_num(variable, nan=0)

        if month == start_month:
            X_matrix = variable
        else:
            X_matrix = np.concatenate((X_matrix, variable), axis=0)

    CO2_change = h_matrix @ X_matrix
    result_df[f'{data_name}'] = CO2_change
result_df.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_GPPobservations.csv', encoding='utf-8', index=False)


# inversions's prior from GCB2023
inversion_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
result_df = pd.DataFrame()
for data_name in inversion_names:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)
        
        # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
        # same order as in the cell_id_table.csv
        nee_vec = subset_30N_90N(read_inversions_prior(data_name, 'prior_flux_land', year, month)).values.flatten()[land_cellnum_list] #unit: PgC/m2/yr
        nee = nee_vec*1e15/12*1e6/365/24/3600 #convert unit to μmol m-2 s-1
        
        if np.isnan(nee).all():
            NEE = nee
        else:
            NEE = np.nan_to_num(nee, nan=0)

        if month == start_month:
            X_matrix = NEE
        else:
            X_matrix = np.concatenate((X_matrix, NEE), axis=0)

    CO2_change = h_matrix @ X_matrix
    result_df[f'{data_name}'] = CO2_change
result_df.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_inversions-prior.csv', encoding='utf-8', index=False)