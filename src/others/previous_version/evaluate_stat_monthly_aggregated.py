# calculate statistics (e.g., Pearson correlation, slope, intercept) of NEE against Arctic-Capairborne measurements
# aggregated CO2 enhancement to monthly and remove sub-monthly variations, to check if (1) cor of CARVE 2012 improves; (2) slope closer to 1
# modified from evaluate_stat.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr

import os
os.chdir('/resnick/groups/carnegie_poc/jwen2/ABoVE/src')
from functions import get_campaign_info

year = 2013 # 2012 2013 2014 2017

start_month, end_month, campaign_name = get_campaign_info(year)


# read table
df_airborne = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/ABoVE_{year}_{campaign_name}_airborne_change.csv')
df_influence = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')
df_fossil_fire = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/ABoVE_{year}_{campaign_name}_airborne_fossil_fire.csv')

# apply filter
local_hour = pd.to_datetime(df_airborne['footprint_time_AKT'], utc=True).dt.tz_convert('America/Anchorage').dt.hour

# filters for airborne observations
lc = 'alllc' # alllc forestshrub tundra
if lc == 'alllc':
    lc_filestr = ''
    mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
    # (local_hour.isin([13, 14, 15, 16])) &
    (df_influence['ABoVE_influence_fraction'] > 0.5) &
    (df_influence['ocean_influence_fraction'] < 0.3) &
    # (df_influence['ABoVE_land_influence_fraction'] > 0.5)) and
    (df_airborne['CO2_change'] < 30) &
    (df_airborne['CO_change'] < 40))[0]
elif lc == 'forestshrub':
    lc_filestr = '_forestshrub'
    mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
    # (local_hour.isin([13, 14, 15, 16])) &
    (df_influence['ABoVE_influence_fraction'] > 0.5) &
    (df_influence['ocean_influence_fraction'] < 0.3) &
    # (df_influence['ABoVE_land_influence_fraction'] > 0.5)) and
    (df_airborne['CO2_change'] < 30) &
    (df_airborne['CO_change'] < 40) &
    ((df_influence['forest_influence'] + df_influence['shrub_influence']) / df_influence['total_influence'] > 0.5))[0]
elif lc == 'tundra':
    lc_filestr = '_tundra'
    mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
    # (local_hour.isin([13, 14, 15, 16])) &
    (df_influence['ABoVE_influence_fraction'] > 0.5) &
    (df_influence['ocean_influence_fraction'] < 0.3) &
    # (df_influence['ABoVE_land_influence_fraction'] > 0.5)) and
    (df_airborne['CO2_change'] < 30) &
    (df_airborne['CO_change'] < 40) &
    (df_influence['tundra_influence'] / df_influence['total_influence'] > 0.5))[0]

# calculate statistics for different models
for model_type in ['TRENDYv11', 'TRENDYv9', 'inversions', 'reference', 'regression']:
    
    if model_type == 'TRENDYv11':
        model_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
        df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/ABoVE_{year}_{campaign_name}_airborne_TRENDYv11.csv')
        fig = plt.figure(figsize=(30,30))
    
    elif model_type == 'TRENDYv9':
        model_names = ['CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT']
        df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/ABoVE_{year}_{campaign_name}_airborne_TRENDYv9.csv')
        fig = plt.figure(figsize=(30,25))
    
    elif model_type == 'inversions':
        # inversions from GCB2023
        model_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
        df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/ABoVE_{year}_{campaign_name}_airborne_inversions.csv')
        # mask columns if all values are zero - for models with no data
        df_model_columns = df_model.columns
        df_model_mask = (df_model[df_model_columns].eq(0).all(axis=0))
        df_model[df_model_columns[df_model_mask]] = np.nan
        fig = plt.figure(figsize=(30,30))
        fig = plt.figure(figsize=(30,30))

    elif model_type == 'reference':
        # remote sensing reference
        model_names = ['APAR', 'PAR', 'FPAR', 'LAI']
        df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/ABoVE_{year}_{campaign_name}_airborne_reference.csv')
        fig = plt.figure(figsize=(30,5))

    elif model_type == 'regression':
        # regression models
        model_names = ['CRU', 'CRUxLC', 'CRUxMonth', 'Month', 'MonthxLC']
        df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/ABoVE_{year}_{campaign_name}_airborne_regression.csv')
        fig = plt.figure(figsize=(30,10))


    df = pd.concat((df_airborne, df_fossil_fire, df_model), axis=1)
    df = df.loc[mask_id]
    df['month'] = pd.to_datetime(df_airborne['footprint_time_AKT'], utc=True).dt.tz_convert('America/Anchorage').dt.month
    selected_column_names = ['CO2_change'] + df_fossil_fire.columns.tolist() + df_model.columns.tolist()
    df = df.groupby(by=df.month)[selected_column_names].mean()

    subplot_id = 0
    for model_name in model_names:
        subplot_id += 1

        plt.subplot(np.ceil(len(model_names)/4).astype(int),4,subplot_id,aspect='equal')

        y = df['CO2_change'].values - df['fossil_CO2_change'] - df['fire_CO2_change']
        x = df[f'{model_name}_CO2_change'] # should I add fire emissions for inversions?
        
        if x.isna().all():
            cor = np.nan; slope = np.nan; intercept = np.nan; r2_1 = np.nan; r2_2 = np.nan
            
        else:
            cor, _ = pearsonr(y, x)

            X = sm.add_constant(x)
            model = sm.OLS(y,X)
            results = model.fit()
            results.params
            slope = results.params[1]
            intercept = results.params[0]
            results.summary()
            y_pred = np.matmul(X, results.params)

            r2_1 = 1 - np.sum((y-x)**2)/np.sum((y-np.mean(y))**2) # calculation 1
            r2_2 = results.rsquared
            # r2_2 = 1 - np.sum((y-y_pred)**2)/np.sum((y-np.mean(y))**2) # same as results.rsquared or corr**2

        plt.scatter(x, y, s=60, alpha=0.7, edgecolors="k")

        fitting_df0 = pd.DataFrame([[model_name, cor, slope, intercept, r2_1, r2_2]], 
                                columns=['model_name', 'cor', 'slope', 'intercept', 'r2_1', 'r2_2'])
        if subplot_id == 1:
            fitting_df = fitting_df0
        else:
            fitting_df = pd.concat([fitting_df, fitting_df0]) 

        
        # Plot 1:1 line
        x_line = np.arange(-30,30,0.1)
        plt.plot(x_line, x_line, color='black', linestyle='dashed')
        y_line = x_line * slope + intercept
        plt.plot(x_line, y_line, color='red', linestyle='dashed')
        plt.annotate(r"$Cor$ = {:.2f}".format(cor), (-28, 25), fontsize=20)
        plt.annotate('y={:.2f}x+{:.2f}'.format(slope,intercept), (-28, 21), fontsize=20)

        plt.xlim(-30, 30)
        plt.ylim(-30, 30)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        # plt.xlabel('Observed CO2 change (removing fossil and fire emissions)', fontsize=20)
        # plt.ylabel('Modeled CO2 change due to NEE', fontsize=15)
        plt.title(model_name, fontsize=40)

    fig.text(0.5, 0.07, 'Modeled CO2 change due to NEE (ppm)', fontsize=40, ha='center')
    fig.text(0.08, 0.5, 'Observed CO2 change (ppm)', fontsize=40, va='center', rotation='vertical') #(removing fossil fuel and fire emissions) 

    plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/evaluation_stat_{campaign_name}_{year}_{model_type}{lc_filestr}_scatterplot_monthly_aggregated.png', dpi=100, bbox_inches='tight')
    plt.show()

    fitting_df.to_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/evaluation_stat_{campaign_name}_{year}_{model_type}{lc_filestr}_monthly_aggregated.csv', encoding='utf-8', index=False)

