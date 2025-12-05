'''
modify the magnitude and seasonal cycle of TRENDY component fluxes
to check how it improves the correlation
reference: observations from inversions and GOSIF-GPP
'''

import numpy as np
import pandas as pd
import os
os.chdir('/resnick/groups/carnegie_poc/jwen2/ABoVE/src')
from functions import get_campaign_info
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

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
        df_airborne = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/ABoVE_{year}_{campaign_name}_airborne_change.csv')
        df_influence = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

        # filters for airborne observations
        mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
            # (local_hour.isin([13, 14, 15, 16])) &
            (df_influence['ABoVE_influence_fraction'] > 0.5) &
            (df_influence['ocean_influence_fraction'] < 0.3) &
            # (df_influence['ABoVE_land_influence_fraction'] > 0.5)) and
            (df_airborne['CO2_change'] < 30) &
            (df_airborne['CO_change'] < 40))[0]
        
        # influence from fossil and fire emissions
        df_fossil_fire = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/ABoVE_{year}_{campaign_name}_airborne_fossil_fire.csv')

        # derive CO2 drawdown/enhancement from fossil and fire emissions
        y0 = df_airborne['CO2_change'].values - df_fossil_fire['fossil_CO2_change'] - df_fossil_fire['fire_CO2_change']
        y_year = y0.loc[mask_id]
        result_df_NEE_year[f'CO2_change_obs'] = y_year

        
        ''' calculate transported NEE seasonal cycle '''
        for month in np.arange(start_month, end_month+1):
            # print(year, month)

            # read files of CO2 change caused by a spatially uniform flux for each footprint and each month
            filename = f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/regression_covariates/constant_{year}_{month}.csv'
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

''' make plots for unmodified data '''
def plot_unmodified_performance(fitting_df_TRENDYv11_sorted):
    fig, ax = plt.subplots(figsize=(7,6))
    plt.scatter(fitting_df_TRENDYv11_sorted['cor'], fitting_df_TRENDYv11_sorted['model_name'], marker='o', color=fitting_df_TRENDYv11_sorted['color'], s=50)
    plt.xlim([-0.2, 0.85])
    plt.xlabel(r'Correlation with observed CO$_{2}$ enhancement', fontsize=18)
    plt.xticks(ticks=np.arange(-0.2, 0.9, 0.1), fontsize=15) #np.arange(-0.2, 1, 0.2), 
    plt.yticks(fontsize=15)
    colors = fitting_df_TRENDYv11_sorted['color'].values.tolist()
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)

''' make plots for modified data '''
def plot_modified_performance(cor_modified_case, fitting_df_TRENDYv11_sorted):
    cor_modified_case_median = cor_modified_case.median(axis=0)
    cor_modified_case_min = cor_modified_case.min(axis=0)
    cor_modified_case_max = cor_modified_case.max(axis=0)
    plt.scatter(cor_modified_case_median, fitting_df_TRENDYv11_sorted['model_name'], marker='s', color='black',facecolor='none', s=50)
    plt.errorbar((cor_modified_case_max+cor_modified_case_min)/2, fitting_df_TRENDYv11_sorted['model_name'], xerr=(cor_modified_case_max-cor_modified_case_min)/2, ecolor='black', fmt='none')


lcname = 'alllc' #alllc forest shrub tundra
lc_filestr = ''
weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'
dir0 = '/resnick/groups/carnegie_poc/jwen2/ABoVE/result/modify_NEE/'

# model performance with original seasonal cycle
fitting_df_TRENDYv11_unscaled_only_seasonal = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_TRENDYv11_only_seasonal.csv')
fitting_df_TRENDYv11_unscaled_only_seasonal = fitting_df_TRENDYv11_unscaled_only_seasonal.loc[~fitting_df_TRENDYv11_unscaled_only_seasonal['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted = fitting_df_TRENDYv11_unscaled_only_seasonal.sort_values('cor')
high_model_subset = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['cor']>0.63, 'model_name'].tolist()
low_model_subset = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['cor']<0.63, 'model_name'].tolist()

# colors for making plots
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(high_model_subset),'color'] = '#5986cb'
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(low_model_subset),'color'] = '#e57f3f'

# read original simulated carbon fluxes
seasonal_df_TRENDYv11NEE = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_TRENDYv11_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11GPP = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_TRENDYv11GPP_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Ra = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_TRENDYv11Ra_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Rh = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_TRENDYv11Rh_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Reco = seasonal_df_TRENDYv11Ra + seasonal_df_TRENDYv11Rh

# read observations
seasonal_df_inversionsNEE = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_inversionsNEE_{regionname}_{lcname}_{weightname}.csv")
inversion_names = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE'] # excluding models without CARVE coverage
seasonal_df_inversionsNEE = seasonal_df_inversionsNEE[inversion_names]
seasonal_df_GOSIFGPP0 = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_GPPobservations_{regionname}_{lcname}_{weightname}.csv")[['GOSIF-GPP']]

# repeat GPP
seasonal_df_GOSIFGPP  = seasonal_df_inversionsNEE.copy()
for model_name in inversion_names:
    seasonal_df_GOSIFGPP[model_name] = seasonal_df_GOSIFGPP0['GOSIF-GPP']

# calculate Reco
seasonal_df_obsReco = seasonal_df_inversionsNEE + seasonal_df_GOSIFGPP

# only select growing seasons (Apr-Nov)
seasonal_df_TRENDYv11NEE = seasonal_df_TRENDYv11NEE.loc[3:10]
seasonal_df_TRENDYv11GPP = seasonal_df_TRENDYv11GPP.loc[3:10]
seasonal_df_TRENDYv11Reco = seasonal_df_TRENDYv11Reco.loc[3:10]

seasonal_df_inversionsNEE = seasonal_df_inversionsNEE.loc[3:10]
seasonal_df_GOSIFGPP = seasonal_df_GOSIFGPP.loc[3:10]
seasonal_df_obsReco = seasonal_df_obsReco.loc[3:10]

''' compare share of component fluxes '''
annual_sum_GPP = seasonal_df_TRENDYv11GPP.sum(axis=0)
annual_sum_Reco = seasonal_df_TRENDYv11Reco.sum(axis=0)
annual_sum_NEE = seasonal_df_TRENDYv11NEE.sum(axis=0)

annual_sum_GPP_obs = seasonal_df_GOSIFGPP.sum(axis=0)
annual_sum_NEE_obs = seasonal_df_inversionsNEE.sum(axis=0)
annual_sum_Reco_obs = seasonal_df_obsReco.sum(axis=0)
# annual_sum_GPP_obs = annual_sum_Reco_obs - annual_sum_NEE_obs

# plot a stacked barplot
# unsorted
# x = seasonal_df_TRENDYv11NEE.columns
# y1 = annual_sum_Ra / annual_sum_GPP
# y2 = annual_sum_Rh / annual_sum_GPP
# y3 = -annual_sum_NEE / annual_sum_GPP

# in a order of model performance
model_sorted = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'][::-1]
x = model_sorted
y1 = annual_sum_Reco[model_sorted] / annual_sum_GPP[model_sorted]
y2 = -annual_sum_NEE[model_sorted] / annual_sum_GPP[model_sorted]
x_obs = inversion_names
y1_obs = annual_sum_Reco_obs[inversion_names] / annual_sum_GPP_obs[inversion_names]
y2_obs = -annual_sum_NEE_obs[inversion_names] / annual_sum_GPP_obs[inversion_names]

# make plots
fig, ax = plt.subplots(figsize=(8, 4))
plt.bar(x, y1, color='b', label='Reco')
plt.bar(x, y2, bottom=y1, color='y', label='NEE')
plt.bar(x_obs, y1_obs, color='b')
plt.bar(x_obs, y2_obs, bottom=y1_obs, color='y')
plt.ylabel("Proportion in GPP", fontsize=14)
plt.xticks(rotation=90)
ax.set_yticks(np.arange(0, 1.2, 0.2), ['0%', '20%', '40%', '60%', '80%', '100%'])
plt.legend()
plt.show()



''' modify NEE '''

'''case 1: modify the share of component fluxes'''
cor_modified_case1 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Reco_model = seasonal_df_TRENDYv11Reco[model_name]

    cor_list = []
    for model_name_ref in inversion_names:

        seasonal_GPP_ref = seasonal_df_GOSIFGPP[model_name_ref]
        seasonal_Reco_ref = seasonal_df_obsReco[model_name_ref]

        # modify magnitude/seasonality of GPP, Reco
        seasonal_GPP_model_modified = modify_magnitude(seasonal_GPP_model, seasonal_GPP_ref)
        seasonal_Reco_model_modified = modify_magnitude(seasonal_Reco_model, seasonal_Reco_ref)
        seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)
    
    cor_modified_case1 = pd.concat((cor_modified_case1, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case1.to_csv(f"{dir0}cor_modified_case1_obs.csv", encoding='utf-8', index=False)

# plot
plot_unmodified_performance(fitting_df_TRENDYv11_unscaled_only_seasonal_sorted)
plot_modified_performance(cor_modified_case1, fitting_df_TRENDYv11_unscaled_only_seasonal_sorted)


'''case 2: modify the seasonality of GPP'''
cor_modified_case2 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Reco_model = seasonal_df_TRENDYv11Reco[model_name]

    cor_list = []
    for model_name_ref in inversion_names:

        seasonal_GPP_ref = seasonal_df_GOSIFGPP[model_name_ref]
        seasonal_Reco_ref = seasonal_df_obsReco[model_name_ref]

        # modify magnitude/seasonality of GPP, Reco
        seasonal_GPP_model_modified = modify_seasonality(seasonal_GPP_model, seasonal_GPP_ref)
        seasonal_Reco_model_modified = seasonal_Reco_model
        seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)
    
    cor_modified_case2 = pd.concat((cor_modified_case2, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case2.to_csv(f"{dir0}cor_modified_case2_obs.csv", encoding='utf-8', index=False)

# plot
plot_unmodified_performance(fitting_df_TRENDYv11_unscaled_only_seasonal_sorted)
plot_modified_performance(cor_modified_case2, fitting_df_TRENDYv11_unscaled_only_seasonal_sorted)


'''case 3: modify the seasonality of Reco'''
cor_modified_case3 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Reco_model = seasonal_df_TRENDYv11Reco[model_name]

    cor_list = []
    for model_name_ref in inversion_names:

        seasonal_GPP_ref = seasonal_df_GOSIFGPP[model_name_ref]
        seasonal_Reco_ref = seasonal_df_obsReco[model_name_ref]

        # modify magnitude/seasonality of GPP, Reco
        seasonal_GPP_model_modified = seasonal_GPP_model
        seasonal_Reco_model_modified = modify_seasonality(seasonal_Reco_model, seasonal_Reco_ref)
        seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)
    
    cor_modified_case3 = pd.concat((cor_modified_case3, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case3.to_csv(f"{dir0}cor_modified_case3_obs.csv", encoding='utf-8', index=False)

# plot
plot_unmodified_performance(fitting_df_TRENDYv11_unscaled_only_seasonal_sorted)
plot_modified_performance(cor_modified_case3, fitting_df_TRENDYv11_unscaled_only_seasonal_sorted)


'''case 4: modify the share of component fluxes and seasonality of GPP'''
cor_modified_case4 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Reco_model = seasonal_df_TRENDYv11Reco[model_name]

    cor_list = []
    for model_name_ref in inversion_names:

        seasonal_GPP_ref = seasonal_df_GOSIFGPP[model_name_ref]
        seasonal_Reco_ref = seasonal_df_obsReco[model_name_ref]

        # modify magnitude/seasonality of GPP, Reco
        seasonal_GPP_model_modified = modify_seasonality(modify_magnitude(seasonal_GPP_model, seasonal_GPP_ref), seasonal_GPP_ref)
        seasonal_Reco_model_modified = modify_magnitude(seasonal_Reco_model, seasonal_Reco_ref)
        seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)
    
    cor_modified_case4 = pd.concat((cor_modified_case4, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case4.to_csv(f"{dir0}cor_modified_case4_obs.csv", encoding='utf-8', index=False)

# plot
plot_unmodified_performance(fitting_df_TRENDYv11_unscaled_only_seasonal_sorted)
plot_modified_performance(cor_modified_case4, fitting_df_TRENDYv11_unscaled_only_seasonal_sorted)



'''case 5: modify the share of component fluxes and seasonality of GPP, Reco'''
cor_modified_case5 = pd.DataFrame()
for model_name in low_model_subset + high_model_subset:

    seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
    seasonal_Reco_model = seasonal_df_TRENDYv11Reco[model_name]

    cor_list = []
    for model_name_ref in inversion_names:

        seasonal_GPP_ref = seasonal_df_GOSIFGPP[model_name_ref]
        seasonal_Reco_ref = seasonal_df_obsReco[model_name_ref]

        # modify magnitude/seasonality of GPP, Reco
        seasonal_GPP_model_modified = modify_seasonality(modify_magnitude(seasonal_GPP_model, seasonal_GPP_ref), seasonal_GPP_ref)
        seasonal_Reco_model_modified = modify_seasonality(modify_magnitude(seasonal_Reco_model, seasonal_Reco_ref), seasonal_Reco_ref)
        seasonal_NEE_model_modified = seasonal_Reco_model_modified - seasonal_GPP_model_modified

        # calculate correlation for modified NEE
        cor = evaluate_seasonal_cycle_cor(seasonal_NEE_model_modified)
        cor_list.append(cor)
        print(cor)
    
    cor_modified_case5 = pd.concat((cor_modified_case5, pd.DataFrame(cor_list, columns=[model_name])), axis=1)

cor_modified_case5.to_csv(f"{dir0}cor_modified_case5_obs.csv", encoding='utf-8', index=False)

# plot
plot_unmodified_performance(fitting_df_TRENDYv11_unscaled_only_seasonal_sorted)
plot_modified_performance(cor_modified_case5, fitting_df_TRENDYv11_unscaled_only_seasonal_sorted)



'''summarizing figure'''
cor_modified_case1 = pd.read_csv(f"{dir0}cor_modified_case1_obs.csv")
cor_modified_case2 = pd.read_csv(f"{dir0}cor_modified_case2_obs.csv")
cor_modified_case3 = pd.read_csv(f"{dir0}cor_modified_case3_obs.csv")

TRENDY_model_subset = low_model_subset #low_model_subset  low_model_subset + high_model_subset

fig, ax = plt.subplots(figsize=(7,7))
plt.scatter(fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(TRENDY_model_subset),'cor'], 
            np.arange(len(TRENDY_model_subset)), marker='x', 
            color=fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(TRENDY_model_subset),'color'], s=50)
plt.xlim([-0.2, 0.85])
plt.xlabel(r'Correlation with observed CO$_{2}$ enhancement', fontsize=18)
plt.xticks(ticks=np.arange(-0.2, 0.9, 0.1), fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=18)
# ax.tick_params(axis='y', colors='#e57f3f')
ax.set_yticks(np.arange(len(TRENDY_model_subset)), TRENDY_model_subset)
colors = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(TRENDY_model_subset),'color']
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)

cor_modified_case = cor_modified_case1
cor_modified_case = cor_modified_case[TRENDY_model_subset]
cor_modified_case_median = cor_modified_case.median(axis=0)
cor_modified_case_min = cor_modified_case.min(axis=0)
cor_modified_case_max = cor_modified_case.max(axis=0)
plt.scatter(cor_modified_case_median, np.arange(len(TRENDY_model_subset))-0.1, marker='s', color='#1A85FF', s=50, label='Component share')
plt.errorbar((cor_modified_case_max+cor_modified_case_min)/2, np.arange(len(TRENDY_model_subset))-0.1, xerr=(cor_modified_case_max-cor_modified_case_min)/2, ecolor='#1A85FF', fmt='none', alpha=0.3)

cor_modified_case = cor_modified_case2
cor_modified_case = cor_modified_case[TRENDY_model_subset]
cor_modified_case_median = cor_modified_case.median(axis=0)
cor_modified_case_min = cor_modified_case.min(axis=0)
cor_modified_case_max = cor_modified_case.max(axis=0)
plt.scatter(cor_modified_case_median, np.arange(len(TRENDY_model_subset))-0.2, marker='d', color='#40B0A6', s=80, label='GPP seasonality')
plt.errorbar((cor_modified_case_max+cor_modified_case_min)/2, np.arange(len(TRENDY_model_subset))-0.2, xerr=(cor_modified_case_max-cor_modified_case_min)/2, ecolor='#40B0A6', fmt='none', alpha=0.3)

cor_modified_case = cor_modified_case3
cor_modified_case = cor_modified_case[TRENDY_model_subset]
cor_modified_case_median = cor_modified_case.median(axis=0)
cor_modified_case_min = cor_modified_case.min(axis=0)
cor_modified_case_max = cor_modified_case.max(axis=0)
plt.scatter(cor_modified_case_median, np.arange(len(TRENDY_model_subset))-0.3, marker='^', color='#5D3A9B', s=80, label='Reco seasonality')
plt.errorbar((cor_modified_case_max+cor_modified_case_min)/2, np.arange(len(TRENDY_model_subset))-0.3, xerr=(cor_modified_case_max-cor_modified_case_min)/2, ecolor='#5D3A9B', fmt='none', alpha=0.3)


# add reference ribbons
fitting_df_regression_scaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_regression{lc_filestr}.csv')
plt.axvspan(fitting_df_regression_scaled.loc[fitting_df_regression_scaled['model_name']=='constant','cor_CI_low'].values[0], fitting_df_regression_scaled.loc[fitting_df_regression_scaled['model_name']=='constant','cor_CI_high'].values[0], alpha=0.2, color='olive')
fitting_df_reference_scaled_only_seasonal = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_reference_only_seasonal{lc_filestr}.csv')
plt.axvspan(fitting_df_reference_scaled_only_seasonal.loc[fitting_df_reference_scaled_only_seasonal['model_name']=='APAR','cor_CI_low'].values[0], fitting_df_reference_scaled_only_seasonal.loc[fitting_df_reference_scaled_only_seasonal['model_name']=='APAR','cor_CI_high'].values[0], alpha=0.2, color='purple')

plt.legend(fontsize=16)