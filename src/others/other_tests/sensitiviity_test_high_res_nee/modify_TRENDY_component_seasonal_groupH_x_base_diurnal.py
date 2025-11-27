'''
adopted from modify_TRENDY_component_seasonal_groupH.py
still adjust TRENDY component fluxes, but impose X-BASE diurnal cycle when calculating correlation
'''

import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info
from scipy.stats import pearsonr

''' calculate correlation based on the mean seasonal cycle '''
def evaluate_seasonal_cycle_cor_x_base_diurnal(mean_seasonal_cycle):

    # create a dataframe to store results
    result_df_NEE = pd.DataFrame()

    # read data year by year
    for year in [2012, 2013, 2014, 2017]:
        # print(year)

        start_month, end_month, campaign_name = get_campaign_info(year)

        # create a dataframe to store results for each year
        result_df_NEE_year = pd.DataFrame()


        ''' read observations '''
        df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
        df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

        # filters for airborne observations
        mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
            # (local_hour.isin([13, 14, 15, 16])) &
            (df_influence['ABoVE_influence_fraction'] > 0.5) &
            (df_influence['ocean_influence_fraction'] < 0.3) &
            # (df_influence['ABoVE_land_influence_fraction'] > 0.5)) and
            (df_airborne['CO2_change'] < 30) &
            (df_airborne['CO_change'] < 40))[0]
        
        # influence from fossil and fire emissions
        df_fossil = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fossil.csv')
        df_fire = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fire.csv')

        # derive CO2 drawdown/enhancement from fossil and fire emissions
        y0 = df_airborne['CO2_change'].values - df_fossil['odiac2022'] - df_fire['gfed4.1']
        y_year = y0.loc[mask_id]
        result_df_NEE_year[f'CO2_change_obs'] = y_year

        
        ''' calculate transported NEE seasonal cycle '''
        for month in np.arange(start_month, end_month+1):
            # print(year, month)

            # read files of CO2 change caused by a spatially uniform flux for each footprint and each month
            filename = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/regression_covariates/constant_{year}_{month}.csv'
            constant0 = pd.read_csv(filename)
            constant0 = constant0.loc[mask_id]
            
            if month == start_month:
                CO2_change_NEE = constant0 * mean_seasonal_cycle[month-1]
            else:
                CO2_change_NEE += constant0 * mean_seasonal_cycle[month-1]


        # diurnal cycle from X-BASE
        df_x_base_monthly_diurnal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_X-BASE-monthly_diurnal.csv')
        df_x_base_monthly = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_X-BASE-monthly.csv')
        df_x_base_diurnal = df_x_base_monthly_diurnal - df_x_base_monthly
        
        result_df_NEE_year[f'CO2_change_model'] = CO2_change_NEE['constant'] + df_x_base_diurnal['X-BASE'].loc[mask_id]

        # combine all years into one dataframe
        result_df_NEE = pd.concat((result_df_NEE, result_df_NEE_year), axis=0)


    # calculate correlation
    pearson_res = pearsonr(result_df_NEE['CO2_change_obs'], result_df_NEE['CO2_change_model'])
    cor, _ = pearson_res

    return cor


''' calculate annual or growing season sum'''
def calculate_annual_sum (mean_seasonal_cycle):
    return np.sum(mean_seasonal_cycle)

''' modify magnitude of carbon flux while keeping seasonality unchanged'''
def modify_magnitude (mean_seasonal_cycle_model, mean_seasonal_cycle_ref):
    result = mean_seasonal_cycle_model / calculate_annual_sum(mean_seasonal_cycle_model) * calculate_annual_sum(mean_seasonal_cycle_ref)
    return result

''' modify seasonality of carbon flux while keeping annual sum unchanged'''
def modify_seasonality (mean_seasonal_cycle_model, mean_seasonal_cycle_ref):
    result = mean_seasonal_cycle_ref / calculate_annual_sum(mean_seasonal_cycle_ref) * calculate_annual_sum(mean_seasonal_cycle_model)
    return result


lcname = 'alllc' #alllc forest shrub tundra
lc_filestr = ''
weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'
dir0 = '/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/seasonality_adjustment/'

# model performance with original seasonal cycle
fitting_df_TRENDYv11_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11_only_seasonal.csv')
fitting_df_TRENDYv11_unscaled_only_seasonal = fitting_df_TRENDYv11_unscaled_only_seasonal.loc[~fitting_df_TRENDYv11_unscaled_only_seasonal['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted = fitting_df_TRENDYv11_unscaled_only_seasonal.sort_values('cor')
high_model_subset = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['cor']>0.63, 'model_name'].tolist()
low_model_subset = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['cor']<0.63, 'model_name'].tolist()
# high_model_subset = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0'] # exclude IBIS
# low_model_subset = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM']

# colors for making plots
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(high_model_subset),'color'] = '#5986cb'
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(low_model_subset),'color'] = '#e57f3f'

# read original simulated carbon fluxes
seasonal_df_TRENDYv11NEE = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11GPP = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11GPP_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Ra = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Ra_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Rh = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Rh_{regionname}_{lcname}_{weightname}.csv")

# only select growing seasons (Apr-Nov) + Mar (because 2013 needs March data)
seasonal_df_TRENDYv11NEE = seasonal_df_TRENDYv11NEE.loc[2:10]
seasonal_df_TRENDYv11GPP = seasonal_df_TRENDYv11GPP.loc[2:10]
seasonal_df_TRENDYv11Ra = seasonal_df_TRENDYv11Ra.loc[2:10]
seasonal_df_TRENDYv11Rh = seasonal_df_TRENDYv11Rh.loc[2:10]



''' modify NEE '''
'''case 1: modify the seasonality of GPP'''
cor_modified_case1 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Ra_model = seasonal_df_TRENDYv11Ra[model_name]
    seasonal_Rh_model = seasonal_df_TRENDYv11Rh[model_name]

    cor_list = []
    for model_name_ref in high_model_subset:

        seasonal_GPP_ref = seasonal_df_TRENDYv11GPP[model_name_ref]
        seasonal_Ra_ref = seasonal_df_TRENDYv11Ra[model_name_ref]
        seasonal_Rh_ref = seasonal_df_TRENDYv11Rh[model_name_ref]

        # modify magnitude/seasonality of GPP, Ra, Rh
        seasonal_GPP_model_modified = modify_seasonality(seasonal_GPP_model, seasonal_GPP_ref)
        seasonal_Ra_model_modified = seasonal_Ra_model
        seasonal_Rh_model_modified = seasonal_Rh_model
        seasonal_NEE_model_modified = seasonal_Ra_model_modified + seasonal_Rh_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor_x_base_diurnal(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)

    cor_modified_case1 = pd.concat((cor_modified_case1, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case1.to_csv(f"{dir0}cor_modified_GPP_seasonality_groupH_x_base_diurnal.csv", encoding='utf-8', index=False)


'''case 2: modify the seasonality of Ra'''
cor_modified_case2 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Ra_model = seasonal_df_TRENDYv11Ra[model_name]
    seasonal_Rh_model = seasonal_df_TRENDYv11Rh[model_name]

    cor_list = []
    for model_name_ref in high_model_subset:

        seasonal_GPP_ref = seasonal_df_TRENDYv11GPP[model_name_ref]
        seasonal_Ra_ref = seasonal_df_TRENDYv11Ra[model_name_ref]
        seasonal_Rh_ref = seasonal_df_TRENDYv11Rh[model_name_ref]

        # modify magnitude/seasonality of GPP, Ra, Rh
        seasonal_GPP_model_modified = seasonal_GPP_model
        seasonal_Ra_model_modified = modify_seasonality(seasonal_Ra_model, seasonal_Ra_ref)
        seasonal_Rh_model_modified = seasonal_Rh_model
        seasonal_NEE_model_modified = seasonal_Ra_model_modified + seasonal_Rh_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor_x_base_diurnal(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)

    cor_modified_case2 = pd.concat((cor_modified_case2, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case2.to_csv(f"{dir0}cor_modified_Ra_seasonality_groupH_x_base_diurnal.csv", encoding='utf-8', index=False)


'''case 3: modify the seasonality of Rh'''
cor_modified_case3 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Ra_model = seasonal_df_TRENDYv11Ra[model_name]
    seasonal_Rh_model = seasonal_df_TRENDYv11Rh[model_name]

    cor_list = []
    for model_name_ref in high_model_subset:

        seasonal_GPP_ref = seasonal_df_TRENDYv11GPP[model_name_ref]
        seasonal_Ra_ref = seasonal_df_TRENDYv11Ra[model_name_ref]
        seasonal_Rh_ref = seasonal_df_TRENDYv11Rh[model_name_ref]

        # modify magnitude/seasonality of GPP, Ra, Rh
        seasonal_GPP_model_modified = seasonal_GPP_model
        seasonal_Ra_model_modified = seasonal_Ra_model
        seasonal_Rh_model_modified = modify_seasonality(seasonal_Rh_model, seasonal_Rh_ref)
        seasonal_NEE_model_modified = seasonal_Ra_model_modified + seasonal_Rh_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor_x_base_diurnal(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)

    cor_modified_case3 = pd.concat((cor_modified_case3, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case3.to_csv(f"{dir0}cor_modified_Rh_seasonality_groupH_x_base_diurnal.csv", encoding='utf-8', index=False)


'''case 4: modify the relative magnitude of component fluxes'''
cor_modified_case4 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Ra_model = seasonal_df_TRENDYv11Ra[model_name]
    seasonal_Rh_model = seasonal_df_TRENDYv11Rh[model_name]

    cor_list = []
    for model_name_ref in high_model_subset:

        seasonal_GPP_ref = seasonal_df_TRENDYv11GPP[model_name_ref]
        seasonal_Ra_ref = seasonal_df_TRENDYv11Ra[model_name_ref]
        seasonal_Rh_ref = seasonal_df_TRENDYv11Rh[model_name_ref]

        # modify magnitude/seasonality of GPP, Ra, Rh
        seasonal_GPP_model_modified = modify_magnitude(seasonal_GPP_model, seasonal_GPP_ref)
        seasonal_Ra_model_modified = modify_magnitude(seasonal_Ra_model, seasonal_Ra_ref)
        seasonal_Rh_model_modified = modify_magnitude(seasonal_Rh_model, seasonal_Rh_ref)
        seasonal_NEE_model_modified = seasonal_Ra_model_modified + seasonal_Rh_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor_x_base_diurnal(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)
    
    cor_modified_case4 = pd.concat((cor_modified_case4, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case4.to_csv(f"{dir0}cor_modified_relative_proportion_groupH_x_base_diurnal.csv", encoding='utf-8', index=False)


'''case 5: modify seasonality of Reco'''
cor_modified_case5 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Ra_model = seasonal_df_TRENDYv11Ra[model_name]
    seasonal_Rh_model = seasonal_df_TRENDYv11Rh[model_name]
    seasonal_Reco_model = seasonal_Ra_model + seasonal_Rh_model

    cor_list = []
    for model_name_ref in high_model_subset:

        seasonal_GPP_ref = seasonal_df_TRENDYv11GPP[model_name_ref]
        seasonal_Ra_ref = seasonal_df_TRENDYv11Ra[model_name_ref]
        seasonal_Rh_ref = seasonal_df_TRENDYv11Rh[model_name_ref]
        seasonal_Reco_ref = seasonal_Ra_ref + seasonal_Rh_ref

        # modify magnitude/seasonality of GPP, Ra, Rh
        seasonal_GPP_model_modified = seasonal_GPP_model
        seasonal_Reco_model_modified = modify_seasonality(seasonal_Reco_model, seasonal_Reco_ref)
        seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor_x_base_diurnal(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)

    cor_modified_case5 = pd.concat((cor_modified_case5, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case5.to_csv(f"{dir0}cor_modified_Reco_seasonality_groupH_x_base_diurnal.csv", encoding='utf-8', index=False)



'''case 6: modify the relative magnitude of component fluxes and seasonality of GPP'''
cor_modified_case6 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Ra_model = seasonal_df_TRENDYv11Ra[model_name]
    seasonal_Rh_model = seasonal_df_TRENDYv11Rh[model_name]

    cor_list = []
    for model_name_ref in high_model_subset:

        seasonal_GPP_ref = seasonal_df_TRENDYv11GPP[model_name_ref]
        seasonal_Ra_ref = seasonal_df_TRENDYv11Ra[model_name_ref]
        seasonal_Rh_ref = seasonal_df_TRENDYv11Rh[model_name_ref]

        # modify magnitude/seasonality of GPP, Ra, Rh
        seasonal_GPP_model_modified = modify_seasonality(modify_magnitude(seasonal_GPP_model, seasonal_GPP_ref), seasonal_GPP_ref)
        seasonal_Ra_model_modified = modify_magnitude(seasonal_Ra_model, seasonal_Ra_ref)
        seasonal_Rh_model_modified = modify_magnitude(seasonal_Rh_model, seasonal_Rh_ref)
        seasonal_NEE_model_modified = seasonal_Ra_model_modified + seasonal_Rh_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor_x_base_diurnal(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)

    cor_modified_case6 = pd.concat((cor_modified_case6, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case6.to_csv(f"{dir0}cor_modified_relative_proportion_GPP_seasonality_groupH_x_base_diurnal.csv", encoding='utf-8', index=False)



'''case 7: modify the relative proportion of component fluxes and seasonality of GPP, Ra'''
cor_modified_case7 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Ra_model = seasonal_df_TRENDYv11Ra[model_name]
    seasonal_Rh_model = seasonal_df_TRENDYv11Rh[model_name]

    cor_list = []
    for model_name_ref in high_model_subset:

        seasonal_GPP_ref = seasonal_df_TRENDYv11GPP[model_name_ref]
        seasonal_Ra_ref = seasonal_df_TRENDYv11Ra[model_name_ref]
        seasonal_Rh_ref = seasonal_df_TRENDYv11Rh[model_name_ref]

        # modify magnitude/seasonality of GPP, Ra, Rh
        seasonal_GPP_model_modified = modify_seasonality(modify_magnitude(seasonal_GPP_model, seasonal_GPP_ref), seasonal_GPP_ref)
        seasonal_Ra_model_modified = modify_seasonality(modify_magnitude(seasonal_Ra_model, seasonal_Ra_ref), seasonal_Ra_ref)
        seasonal_Rh_model_modified = modify_magnitude(seasonal_Rh_model, seasonal_Rh_ref)
        seasonal_NEE_model_modified = seasonal_Ra_model_modified + seasonal_Rh_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor_x_base_diurnal(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)

    cor_modified_case7 = pd.concat((cor_modified_case7, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case7.to_csv(f"{dir0}cor_modified_relative_proportion_NPP_seasonality_groupH_x_base_diurnal.csv", encoding='utf-8', index=False)
