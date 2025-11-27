'''
convert mean seasonal cycle of surface fields to concentration space
both spatial variation and interannual variation are removed
'''

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info


# TRENDY v11
TRENDYv11_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
seasonal_df_multiyear = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11_ABoVEcore_alllc_unweighted.csv')

for year in [2012, 2013, 2014, 2017]:
    print(year)

    start_month, end_month, campaign_name = get_campaign_info(year)

    result_df_NEE = pd.DataFrame()

    for data_name in TRENDYv11_names:
        print(data_name)

        # read X matrix
        for month in np.arange(start_month, end_month+1):
            print(year, month)

            # read files of CO2 change caused by a spatially uniform flux for each footprint and each month
            filename = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/regression_covariates/constant_{year}_{month}.csv'
            constant0 = pd.read_csv(filename)

            seasonal_df_multiyear_individual = seasonal_df_multiyear[data_name]
            
            if month == start_month:
                CO2_change_NEE = constant0 * seasonal_df_multiyear_individual[month-1]
            else:
                CO2_change_NEE += constant0 * seasonal_df_multiyear_individual[month-1]

        
        result_df_NEE[f'{data_name}'] = CO2_change_NEE

    result_df_NEE.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_TRENDYv11_only_seasonal.csv', encoding='utf-8', index=False)



# inversions from GCB2023
inversion_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
seasonal_df_multiyear = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_inversionsNEE_ABoVEcore_alllc_unweighted.csv')

for year in [2012, 2013, 2014, 2017]:
    print(year)

    start_month, end_month, campaign_name = get_campaign_info(year)

    result_df_NEE = pd.DataFrame()

    for data_name in inversion_names:
        print(data_name)

        # read X matrix
        for month in np.arange(start_month, end_month+1):
            print(year, month)

            # read files of CO2 change caused by a spatially uniform flux for each footprint and each month
            filename = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/regression_covariates/constant_{year}_{month}.csv'
            constant0 = pd.read_csv(filename)

            seasonal_df_multiyear_individual = seasonal_df_multiyear[data_name]
            
            if month == start_month:
                CO2_change_NEE = constant0 * seasonal_df_multiyear_individual[month-1]
            else:
                CO2_change_NEE += constant0 * seasonal_df_multiyear_individual[month-1]

        
        result_df_NEE[f'{data_name}'] = CO2_change_NEE

    result_df_NEE.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_inversionsNEE_only_seasonal.csv', encoding='utf-8', index=False)


# Upscaled EC datasets
UpscaledEC_names = ['X-BASE', 'ABCflux']
seasonal_df_multiyear = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_UpscaledEC_ABoVEcore_alllc_unweighted.csv')

for year in [2012, 2013, 2014, 2017]:
    print(year)

    start_month, end_month, campaign_name = get_campaign_info(year)

    result_df_NEE = pd.DataFrame()

    for data_name in UpscaledEC_names:
        print(data_name)

        # read X matrix
        for month in np.arange(start_month, end_month+1):
            print(year, month)

            # read files of CO2 change caused by a spatially uniform flux for each footprint and each month
            filename = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/regression_covariates/constant_{year}_{month}.csv'
            constant0 = pd.read_csv(filename)

            seasonal_df_multiyear_individual = seasonal_df_multiyear[data_name]
            
            if month == start_month:
                CO2_change_NEE = constant0 * seasonal_df_multiyear_individual[month-1]
            else:
                CO2_change_NEE += constant0 * seasonal_df_multiyear_individual[month-1]

        
        result_df_NEE[f'{data_name}'] = CO2_change_NEE

    result_df_NEE.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_UpscaledEC_only_seasonal.csv', encoding='utf-8', index=False)


# reference
reference_names = ['APAR', 'PAR', 'FPAR', 'LAI', 'NDVI', 'EVI', 'GOME2_SIF']
seasonal_df_multiyear = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_reference_ABoVEcore_alllc_unweighted.csv')

for year in [2012, 2013, 2014, 2017]:
    print(year)

    start_month, end_month, campaign_name = get_campaign_info(year)

    result_df_NEE = pd.DataFrame()

    for data_name in reference_names:
        print(data_name)

        # read X matrix
        for month in np.arange(start_month, end_month+1):
            print(year, month)

            # read files of CO2 change caused by a spatially uniform flux for each footprint and each month
            filename = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/regression_covariates/constant_{year}_{month}.csv'
            constant0 = pd.read_csv(filename)

            seasonal_df_multiyear_individual = seasonal_df_multiyear[data_name]
            seasonal_df_multiyear_individual[pd.isna(seasonal_df_multiyear_individual)] = 0
            
            if month == start_month:
                CO2_change_NEE = constant0 * seasonal_df_multiyear_individual[month-1]
            else:
                CO2_change_NEE += constant0 * seasonal_df_multiyear_individual[month-1]

        
        result_df_NEE[f'{data_name}'] = CO2_change_NEE

    result_df_NEE.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_reference_only_seasonal.csv', encoding='utf-8', index=False)


# inversions' prior from GCB2023
inversion_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
seasonal_df_multiyear = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_inversionsNEE-prior_ABoVEcore_alllc_unweighted.csv')

for year in [2012, 2013, 2014, 2017]:
    print(year)

    start_month, end_month, campaign_name = get_campaign_info(year)

    result_df_NEE = pd.DataFrame()

    for data_name in inversion_names:
        print(data_name)

        # read X matrix
        for month in np.arange(start_month, end_month+1):
            print(year, month)

            # read files of CO2 change caused by a spatially uniform flux for each footprint and each month
            filename = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/regression_covariates/constant_{year}_{month}.csv'
            constant0 = pd.read_csv(filename)

            seasonal_df_multiyear_individual = seasonal_df_multiyear[data_name]
            
            if month == start_month:
                CO2_change_NEE = constant0 * seasonal_df_multiyear_individual[month-1]
            else:
                CO2_change_NEE += constant0 * seasonal_df_multiyear_individual[month-1]

        
        result_df_NEE[f'{data_name}'] = CO2_change_NEE

    result_df_NEE.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_inversionsNEE-prior_only_seasonal.csv', encoding='utf-8', index=False)

