'''plot NEE seasonal cycles of inversions, upscaled EC, and TRENDY TBMs'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

selection = True # True False

'''plot NEE'''
ylim1 = -4
ylim2 = 2

for model_type, color, text in [('TRENDYv11', '#d4631d', 'TRENDY v11'), ('UpscaledEC', '#4c8638', 'X-BASE, ABCflux'), ('inversionsNEE', 'black', 'GCB2023 inversions')]: # #56983f

    weightname = 'unweighted' #unweighted weighted
    lcname = 'alllc' #alllc forestshrub forest shrub tundra
    regionname = 'ABoVEcore'

    seasonal_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_{model_type}_{regionname}_{lcname}_{weightname}.csv')
    seasonal_df = seasonal_df.loc[:, ~seasonal_df.columns.isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU'])] ## for models with no coverage of CARVE years
    model_names = seasonal_df.columns.tolist()

    if selection:
        fitting_df_ct = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_{model_type}_background-ct.csv')
        fitting_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_{model_type}.csv')
        # Filter models by correlation, mean bias, and range ratio thresholds
        corr_thr = 0.61
        bias_thr = 0.5
        range_thr = 0.3

        corr = fitting_df['cor']
        bias = fitting_df_ct['mean_bias']
        range_ratio = fitting_df_ct['range_ratio_95_5']

        mask = (corr > corr_thr) & (bias.abs() < bias_thr) & ((range_ratio-1).abs() < range_thr)
        model_names_selected = seasonal_df.columns[mask.values]

        # Keep only filtered models in seasonal_df
        seasonal_df = seasonal_df[model_names_selected]

    fig, ax = plt.subplots(figsize=(6,4))
    for model_name in seasonal_df.columns:
        plt.plot(np.arange(1,13), seasonal_df[model_name], linestyle='-', color=color, alpha=0.8)
        plt.xlabel('Month', fontsize=15)
        plt.ylabel(f'NEE ' + '($\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=15)
        ax.set_xticks(np.arange(1,13))
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(ylim1, ylim2)

    if not selection:
        ax.text(0.02, 0.05, text, transform=ax.transAxes, fontsize=15, color=color, ha='left', va='bottom')
    plt.show()


'''plot NEE components'''
for model_type, color in [('TRENDYv11GPP', '#438382'), ('TRENDYv11Ra', '#c0ac1a'), ('TRENDYv11Rh', '#e573d5')]:

    if model_type == 'TRENDYv11GPP':
        ylim1 = -2
        ylim2 = 10
        varname = 'GPP'
    else:
        ylim1 = -1
        ylim2 = 5
        varname = model_type.replace('TRENDYv11', '')
        varname = varname.replace('Ra', r'R$_a$').replace('Rh', r'R$_h$')
    
    weightname = 'unweighted' #unweighted weighted
    lcname = 'alllc' #alllc forestshrub forest shrub tundra
    regionname = 'ABoVEcore'

    seasonal_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_{model_type}_{regionname}_{lcname}_{weightname}.csv')
    fig, ax = plt.subplots(figsize=(6,4))
    for model_name in seasonal_df.columns:
        if model_name != 'IBIS':
            plt.plot(np.arange(1,13), seasonal_df[model_name], linestyle='-', color=color, alpha=0.8)
            plt.xlabel('Month', fontsize=15)
            plt.ylabel(f'{varname} ' + '($\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=15)
            ax.set_xticks(np.arange(1,13))
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.ylim(ylim1, ylim2)

    plt.show()