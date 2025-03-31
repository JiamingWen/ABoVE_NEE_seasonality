'''
convert variables (e.g., spatially-uniform flux, remote sensing variables, CRU inputs, land cover) 
into concentration space, for regression later
'''

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info, read_remote_sensing, read_cru, read_MODIS_VI, read_GOME2_SIF

year = 2012 # 2012 2013 2014 2017

start_month, end_month, campaign_name = get_campaign_info(year)

# read observations
receptor_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
n_receptor = receptor_df.shape[0]

# mask for land pixels
cell_id_table = pd.read_csv('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/cell_id_table/cell_id_table.csv')
land_cellnum_list = np.where(cell_id_table['land']==1)[0]

dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/regression_covariates/"
if not os.path.exists(dir0):
    os. makedirs(dir0)


for month in np.arange(start_month,end_month+1):
    print(year, month)

    # read H matrix
    h_df = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/H{year}_{month}.txt",
                    sep="\s+", index_col=False, header=None,
                    names=["obs_id", "cell_id", "lat_id","lon_id", "lat", "lon", "val"])
    #  \s+ is the expression for "any amount of whitespace"

    n_cell = 720 * 120
    h_matrix0 = csr_matrix((h_df.val, (h_df.obs_id, h_df.cell_id)),  
                            shape = (n_receptor, n_cell))
    
    h_matrix0_subset = h_matrix0[:, land_cellnum_list]
    del h_matrix0


    # spatially-uniform flux
    variable = np.ones((n_cell))[land_cellnum_list]
    covariate = h_matrix0_subset @ variable
    covariate = pd.DataFrame(covariate, columns=['constant'])
    covariate.to_csv(f'{dir0}constant_{year}_{month}.csv', encoding='utf-8', index=False)


    # remote-sensing variables
    for data_name in ['APAR', 'PAR', 'FPAR', 'LAI', 'NDVI', 'EVI', 'GOME2_SIF']: #'APAR', 'PAR', 'FPAR', 'LAI', 'NDVI', 'EVI', 'GOME2_SIF'
        if data_name == 'PAR':
            variable = read_remote_sensing('par', 'PAR', year, month).values.flatten()[land_cellnum_list]
        elif data_name == 'FPAR':
            variable = read_remote_sensing('fpar', 'Fpar', year, month).values.flatten()[land_cellnum_list]
        elif data_name == 'LAI':
            variable = read_remote_sensing('lai', 'Lai', year, month).values.flatten()[land_cellnum_list]
        elif data_name == 'APAR':
            par_vec = read_remote_sensing('par', 'PAR', year, month).values.flatten()[land_cellnum_list]
            fpar_vec = read_remote_sensing('fpar', 'Fpar', year, month).values.flatten()[land_cellnum_list]
            variable = par_vec * fpar_vec
        elif data_name == 'NDVI':
            variable = read_MODIS_VI('NDVI', year, month).values.flatten()[land_cellnum_list]
            variable = np.nan_to_num(variable, nan=0)
        elif data_name == 'EVI':
            variable = read_MODIS_VI('EVI', year, month).values.flatten()[land_cellnum_list]
            variable = np.nan_to_num(variable, nan=0)
        elif data_name == 'GOME2_SIF':
            variable = read_GOME2_SIF('dcSIF', year, month).values.flatten()[land_cellnum_list]
            variable = np.nan_to_num(variable, nan=0)

        covariate = h_matrix0_subset @ variable
        covariate = pd.DataFrame(covariate, columns=[data_name])
        covariate.to_csv(f'{dir0}{data_name}_{year}_{month}.csv', encoding='utf-8', index=False)
        
        
    # cru variables
    for data_name in ['dswrf', 'pre', 'spfh', 'tmp']: 
        variable = read_cru(data_name, data_name, year, month).values.flatten()[land_cellnum_list]
        variable = np.nan_to_num(variable, nan=0)

        covariate = h_matrix0_subset @ variable
        covariate = pd.DataFrame(covariate, columns=[data_name])
        covariate.to_csv(f'{dir0}{data_name}_{year}_{month}.csv', encoding='utf-8', index=False)

    # land covers
    for lcname, lcid in zip(['forest', 'shrub', 'tundra', 'others'], [[5], [7], [8,9,10], [0,1,2,3,4,6,11,12,13,14,15]]):
        variable = np.zeros((n_cell))
        variable[np.where(cell_id_table['lc'] .isin (lcid))[0]] = 1
        variable = variable[land_cellnum_list]
        covariate = h_matrix0_subset @ variable
        covariate = pd.DataFrame(covariate, columns=['constant'])
        covariate.to_csv(f'{dir0}constant_{lcname}_{year}_{month}.csv', encoding='utf-8', index=False)