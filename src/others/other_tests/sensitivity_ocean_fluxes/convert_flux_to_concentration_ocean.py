'''convert ocean fluxes to the concentration space'''

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
import os
import sys
sys.path.append('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info, read_H_matrix, subset_30N_90N, read_ocean_fluxes

year = 2017 # 2012 2013 2014 2017

start_month, end_month, campaign_name = get_campaign_info(year)

dir_out = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field'

# read observations
receptor_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
n_receptor = receptor_df.shape[0]


# mask for land pixels
cell_id_table = pd.read_csv('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/cell_id_table/cell_id_table.csv')
ocean_cellnum_list = np.where(cell_id_table['land']==0)[0]

# read H matrix
h_matrix = read_H_matrix(year, n_receptor, ocean_cellnum_list)

# ocean fluxes from GCB2024
ocean_flux_names = ['CMEMS-LSCE-FFNN', 'CSIR-ML6', 'JENA-MLS', 'JMA-MLR', 'LDEO-HPD', 'NIES-ML3', 'OceanSODA-ETHZv2', 'UoEX-UEPFFNU', 'VLIZ-SOMFFN']
result_df = pd.DataFrame()
for data_name in ocean_flux_names:
    print(data_name)

    # read X matrix
    for month in np.arange(start_month, end_month+1):
        print(year, month)
        
        # by lat starting from 30.25N (-179.75, ..., 179.75), then 30.75N
        # same order as in the cell_id_table.csv
        nee_vec = subset_30N_90N(read_ocean_fluxes(data_name, year, month)).values.flatten()[ocean_cellnum_list] #unit: mol/m2/s
        nee = nee_vec*1e6 #convert unit to μmol m-2 s-1

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
result_df.to_csv(f'{dir_out}/ABoVE_{year}_{campaign_name}_airborne_ocean.csv', encoding='utf-8', index=False)