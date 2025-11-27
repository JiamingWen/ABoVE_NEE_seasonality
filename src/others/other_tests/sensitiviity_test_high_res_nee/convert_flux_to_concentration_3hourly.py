'''convert 3-hourly fluxes to the concentration space'''

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
import os
import sys
sys.path.append('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
import utils
from functions import get_campaign_info, read_H_matrix_3hourly, subset_30N_90N, read_CT_NOAA_3hourly, read_CTE_3hourly


year = 2017 # 2012 2013 2014 2017

campaign_name = get_campaign_info(year)[2]
config = utils.getConfig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/config/config_{campaign_name}{year}_3hourly.ini')

# read observations
receptor_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
n_receptor = receptor_df.shape[0]


# mask for land pixels
cell_id_table = pd.read_csv('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/cell_id_table/cell_id_table.csv')
land_cellnum_list = np.where(cell_id_table['land']==1)[0]


# read H matrix
h_matrix = read_H_matrix_3hourly(year, n_receptor, land_cellnum_list)


# 3-hourly CT-NOAA
result_df = pd.DataFrame()
X_matrix = None
for ntimestep in np.arange(0, config["ntimesteps"]):
    print(ntimestep)
    
    timestep = config["sdate"] + ntimestep * config["timestep"]
    NEE_vec = subset_30N_90N(read_CT_NOAA_3hourly(timestep.year, timestep.month, timestep.day, timestep.hour)).values.flatten()[land_cellnum_list] #unit: mol m-2 s-1
    NEE = NEE_vec * 1e6 #convert unit to μmol m-2 s-1
    
    if ntimestep == 0:
        X_matrix = NEE
    else:
        X_matrix = np.concatenate((X_matrix, NEE), axis=0)

CO2_change = h_matrix @ X_matrix
result_df[f'CT-NOAA'] = CO2_change
result_df.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_CT-NOAA-3hourly.csv', encoding='utf-8', index=False)


# 3-hourly CTE
result_df = pd.DataFrame()
X_matrix = None
for ntimestep in np.arange(0, config["ntimesteps"]):
    print(ntimestep)
    
    timestep = config["sdate"] + ntimestep * config["timestep"]
    NEE_vec = subset_30N_90N(read_CTE_3hourly(timestep.year, timestep.month, timestep.day, timestep.hour)).values.flatten()[land_cellnum_list] #unit: mol m-2 s-1
    NEE = NEE_vec * 1e6 #convert unit to μmol m-2 s-1
    
    if ntimestep == 0:
        X_matrix = NEE
    else:
        X_matrix = np.concatenate((X_matrix, NEE), axis=0)

CO2_change = h_matrix @ X_matrix
result_df[f'CTE'] = CO2_change
result_df.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_CTE-3hourly.csv', encoding='utf-8', index=False)

