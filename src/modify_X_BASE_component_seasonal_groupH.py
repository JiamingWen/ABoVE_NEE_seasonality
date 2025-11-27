'''
modify the magnitude and seasonal cycle of X-BASE component fluxes, 
check how it improves the correlation
'''

import numpy as np
import pandas as pd
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info
from scipy.stats import pearsonr

''' calculate correlation based on the mean seasonal cycle '''
def evaluate_seasonal_cycle_cor(mean_seasonal_cycle):

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

        result_df_NEE_year[f'CO2_change_model'] = CO2_change_NEE

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
dir0 = '/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonality_adjustment/'

# model performance statistics
fitting_df_TRENDYv11_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11_only_seasonal.csv')
fitting_df_TRENDYv11_unscaled_only_seasonal = fitting_df_TRENDYv11_unscaled_only_seasonal.loc[~fitting_df_TRENDYv11_unscaled_only_seasonal['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted = fitting_df_TRENDYv11_unscaled_only_seasonal.sort_values('cor')
high_model_subset = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['cor']>0.63, 'model_name'].tolist()

fitting_df_UpscaledEC_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC_only_seasonal.csv')


# read original simulated carbon fluxes
# X-BASE
seasonal_df_NEE = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_UpscaledEC_{regionname}_{lcname}_{weightname}.csv")[['X-BASE']]
seasonal_df_GPP = pd.read_csv(f"//central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_UpscaledEC_GPP_{regionname}_{lcname}_{weightname}.csv")[['X-BASE']]
seasonal_df_Reco = seasonal_df_NEE + seasonal_df_GPP

# TRENDY
seasonal_df_TRENDYv11NEE = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11GPP = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11GPP_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Reco = seasonal_df_TRENDYv11NEE + seasonal_df_TRENDYv11GPP

# ABCflux
seasonal_df_ABCfluxNEE = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_UpscaledEC_{regionname}_{lcname}_{weightname}.csv")[['ABCflux']]
seasonal_df_ABCfluxGPP = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_UpscaledEC_GPP_{regionname}_{lcname}_{weightname}.csv")[['ABCflux']]
# seasonal_df_ABCfluxReco = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_Recoobservations_{regionname}_{lcname}_{weightname}.csv")[['ABCflux']]
seasonal_df_ABCfluxReco = seasonal_df_ABCfluxNEE + seasonal_df_ABCfluxGPP

# only select growing seasons (Apr-Nov) + Mar (because 2013 needs March data)
seasonal_df_NEE = seasonal_df_NEE.loc[2:10]
seasonal_df_GPP = seasonal_df_GPP.loc[2:10]
seasonal_df_Reco = seasonal_df_Reco.loc[2:10]

seasonal_df_TRENDYv11NEE = seasonal_df_TRENDYv11NEE.loc[2:10]
seasonal_df_TRENDYv11GPP = seasonal_df_TRENDYv11GPP.loc[2:10]
seasonal_df_TRENDYv11Reco = seasonal_df_TRENDYv11Reco.loc[2:10]



'''case 1: modify the seasonality of GPP'''
seasonal_GPP_model = seasonal_df_GPP['X-BASE']
seasonal_Reco_model = seasonal_df_Reco['X-BASE']

cor_modified_case_GPP = pd.DataFrame()
cor_list = []
for model_name_ref in high_model_subset:

    seasonal_GPP_ref = seasonal_df_TRENDYv11GPP[model_name_ref]
    seasonal_Reco_ref = seasonal_df_TRENDYv11Reco[model_name_ref]

    # modify magnitude/seasonality of GPP, Reco
    seasonal_GPP_model_modified = modify_seasonality(seasonal_GPP_model, seasonal_GPP_ref)
    seasonal_Reco_model_modified = seasonal_Reco_model
    seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified

    # calculate correlation for modified NEE
    cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
    cor_list.append(cor)
    # print(cor)

cor_modified_case_GPP = pd.concat((cor_modified_case_GPP, pd.DataFrame(cor_list, columns=['X-BASE'])), axis=1)
cor_modified_case_GPP.to_csv(f"{dir0}cor_modified_GPP_seasonality_X-BASE_groupH.csv", encoding='utf-8', index=False)
np.median(cor_modified_case_GPP) # 0.5662969750641214

# using ABCflux as reference
seasonal_GPP_model_modified = modify_seasonality(seasonal_GPP_model, seasonal_df_ABCfluxGPP['ABCflux'])
seasonal_Reco_model_modified = seasonal_Reco_model
seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified
cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
print(cor) # 0.5886706497393405


'''case 2: modify the seasonality of Reco'''
seasonal_GPP_model = seasonal_df_GPP['X-BASE']
seasonal_Reco_model = seasonal_df_Reco['X-BASE']

cor_modified_case_Reco = pd.DataFrame()
cor_list = []
for model_name_ref in high_model_subset:

    seasonal_GPP_ref = seasonal_df_TRENDYv11GPP[model_name_ref]
    seasonal_Reco_ref = seasonal_df_TRENDYv11Reco[model_name_ref]

    # modify magnitude/seasonality of GPP, Reco
    seasonal_GPP_model_modified = seasonal_GPP_model
    seasonal_Reco_model_modified = modify_seasonality(seasonal_Reco_model, seasonal_Reco_ref) 
    seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified

    # calculate correlation for modified NEE
    cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
    cor_list.append(cor)
    # print(cor)

cor_modified_case_Reco = pd.concat((cor_modified_case_Reco, pd.DataFrame(cor_list, columns=['X-BASE'])), axis=1)
cor_modified_case_Reco.to_csv(f"{dir0}cor_modified_Reco_seasonality_X-BASE_groupH.csv", encoding='utf-8', index=False)
np.median(cor_modified_case_Reco) # 0.6430052824915442

# using ABCflux as reference
seasonal_GPP_model_modified = seasonal_GPP_model
seasonal_Reco_model_modified = modify_seasonality(seasonal_Reco_model, seasonal_df_ABCfluxReco['ABCflux']) 
seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified
cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
print(cor) # 0.5860969504080284 - much lower than using higher-cor TBMs as reference


'''case 3: modify the relative magnitude of component fluxes'''
seasonal_GPP_model = seasonal_df_GPP['X-BASE']
seasonal_Reco_model = seasonal_df_Reco['X-BASE']

cor_modified_case_Magnitude = pd.DataFrame()
cor_list = []
for model_name_ref in high_model_subset:

    seasonal_GPP_ref = seasonal_df_TRENDYv11GPP[model_name_ref]
    seasonal_Reco_ref = seasonal_df_TRENDYv11Reco[model_name_ref]

    # modify magnitude/seasonality of GPP, Reco
    seasonal_GPP_model_modified = modify_magnitude(seasonal_GPP_model, seasonal_GPP_ref)
    seasonal_Reco_model_modified = modify_magnitude(seasonal_Reco_model, seasonal_Reco_ref) 
    seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified

    # calculate correlation for modified NEE
    cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
    cor_list.append(cor)
    # print(cor)

cor_modified_case_Magnitude = pd.concat((cor_modified_case_Magnitude, pd.DataFrame(cor_list, columns=['X-BASE'])), axis=1)
cor_modified_case_Magnitude.to_csv(f"{dir0}cor_modified_relative_proportion_X-BASE_groupH.csv", encoding='utf-8', index=False)
np.median(cor_modified_case_Magnitude) # 0.5651814444980734

# using ABCflux as reference
seasonal_GPP_model_modified = modify_magnitude(seasonal_GPP_model, seasonal_df_ABCfluxGPP['ABCflux'])
seasonal_Reco_model_modified = modify_magnitude(seasonal_Reco_model, seasonal_df_ABCfluxReco['ABCflux']) 
sum(seasonal_Reco_model) / sum(seasonal_GPP_model) #0.85
sum(seasonal_df_ABCfluxReco['ABCflux']) / sum(seasonal_df_ABCfluxGPP['ABCflux']) # 0.97
seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified
cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
print(cor) # 0.5638286449027919


# try magnitude and seasonality together for ABCflux reference
# magnitude + GPP seasonality
seasonal_GPP_model_modified = modify_seasonality(modify_magnitude(seasonal_GPP_model, seasonal_df_ABCfluxGPP['ABCflux']), seasonal_df_ABCfluxGPP['ABCflux'])
seasonal_Reco_model_modified = modify_magnitude(seasonal_Reco_model, seasonal_df_ABCfluxReco['ABCflux']) 
seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified
cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
print(cor) # 0.586041000816082

# magnitude + Reco seasonality
seasonal_GPP_model_modified = modify_magnitude(seasonal_GPP_model, seasonal_df_ABCfluxGPP['ABCflux'])
seasonal_Reco_model_modified = modify_seasonality(modify_magnitude(seasonal_Reco_model, seasonal_df_ABCfluxReco['ABCflux']) , seasonal_df_ABCfluxReco['ABCflux'])
seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified
cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
print(cor) # 0.6034694635923228