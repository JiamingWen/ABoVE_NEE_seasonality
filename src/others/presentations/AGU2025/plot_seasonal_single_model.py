'''plot the seasonal cycle of NEE and its components for two models, to demonstrate the seasonality adjustment'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ylim1 = -2
ylim2 = 5


model_type = 'TRENDYv11'
model_name = 'CABLE-POP' # ISBA-CTRIP CABLE-POP

weightname = 'unweighted' #unweighted weighted
lcname = 'alllc' #alllc forestshrub forest shrub tundra
regionname = 'ABoVEcore'

''' calculate annual or growing season sum'''
def calculate_annual_sum (mean_seasonal_cycle):
    return np.sum(mean_seasonal_cycle)

''' modify seasonality of carbon flux while keeping annual sum unchanged'''
def modify_seasonality (mean_seasonal_cycle_model, mean_seasonal_cycle_ref):
    result = mean_seasonal_cycle_ref / calculate_annual_sum(mean_seasonal_cycle_ref) * calculate_annual_sum(mean_seasonal_cycle_model)
    return result

for model_type in ['TRENDYv11', 'TRENDYv11GPP', 'TRENDYv11Ra', 'TRENDYv11Rh']:
    if model_type == 'TRENDYv11':
        if model_name == 'ISBA-CTRIP':
            color = '#1367e9'
        else:
            color = '#d4631d'
    elif model_type == 'TRENDYv11GPP':
        color = '#438382'
    elif model_type == 'TRENDYv11Ra':
        color = '#c0ac1a'
    else:
        color = '#e573d5'


    seasonal_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_{model_type}_{regionname}_{lcname}_{weightname}.csv')
    
    fig, ax = plt.subplots(figsize=(4,4))
    plt.plot(np.arange(4,12), seasonal_df[model_name][3:11], linestyle='-', color=color, linewidth=5)
    
    if model_name == 'CABLE-POP':
        if model_type in ['TRENDYv11GPP']: #, 'TRENDYv11Ra', 'TRENDYv11Rh'
            plt.plot(np.arange(4,12), modify_seasonality(seasonal_df[model_name][3:11], seasonal_df['ISBA-CTRIP'][3:11]) , linestyle='--', color=color, linewidth=5)
        elif model_type == 'TRENDYv11':
            # use adjusting GPP as an example
            seasonal_df_gpp = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11GPP_{regionname}_{lcname}_{weightname}.csv')
            seasonal_df_ra = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Ra_{regionname}_{lcname}_{weightname}.csv')
            seasonal_df_rh = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Rh_{regionname}_{lcname}_{weightname}.csv')
            nee_adjusted = seasonal_df_ra[model_name][3:11] + seasonal_df_rh[model_name][3:11] - modify_seasonality(seasonal_df_gpp[model_name][3:11], seasonal_df_gpp['ISBA-CTRIP'][3:11])
            plt.plot(np.arange(4,12), nee_adjusted , linestyle='--', color=color, linewidth=5)

    plt.xlabel('Month', fontsize=15)
    plt.ylabel(f'Carbon fluxes ' + '($\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=15)
    ax.set_xlim(3.5,11.5)
    ax.set_xticks(np.arange(4,12))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(ylim1, ylim2)
    plt.show()
