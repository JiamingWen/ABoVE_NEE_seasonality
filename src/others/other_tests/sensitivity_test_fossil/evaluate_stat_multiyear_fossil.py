'''
sensitivity test - replace ODIAC2022 with GridFED2024 for fossil fuel emissions
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLSResults
import sys
sys.path.append('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info

dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/"
dir1 = f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/regression/"

# whether to filter observations based on land covers they are most sensitive to
lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

# model types
# all types
model_types = ['TRENDYv11', 'inversionsNEE', 'UpscaledEC', 'reference'] #'TRENDYv11', 'inversionsNEE', 'UpscaledEC', 'reference', 'regression', 'TRENDYv11GPP', 'TRENDYv11NPP', 'TRENDYv11Reco', 'UpscaledEC_GPP', 'GPPobservations', 'UpscaledEC_Reco', 'TRENDYv11_only_seasonal', 'inversionsNEE_only_seasonal', 'UpscaledEC_only_seasonal', 'reference_only_seasonal', 'inversionsNEE-prior', 'inversionsNEE-prior_only_seasonal'
# only one variable (i.e., not regression like CRU or LC)
model_types_single = ['TRENDYv11', 'inversionsNEE', 'UpscaledEC', 'reference', 'TRENDYv11GPP', 'TRENDYv11NPP', 'TRENDYv11Reco', 'UpscaledEC_GPP', 'GPPobservations', 'UpscaledEC_Reco', 'TRENDYv11_only_seasonal', 'inversionsNEE_only_seasonal', 'UpscaledEC_only_seasonal', 'reference_only_seasonal', 'inversionsNEE-prior', 'inversionsNEE-prior_only_seasonal']

for model_type in model_types:

    print(model_type)

    if model_type in ['TRENDYv11', 'TRENDYv11GPP', 'TRENDYv11NPP', 'TRENDYv11Reco', 'TRENDYv11_only_seasonal']:
        model_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
        figsize = (20, 20); rownum = 4; colnum = 4; axislabelsize = 25; xlabel_loc = (0.5, 0.07); ylabel_loc = (0.07, 0.5)
    elif model_type in ['inversionsNEE','inversionsNEE_only_seasonal']:
        # model_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
        model_names = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE'] # remove the five satellite-based inversions
        figsize = (20, 14); rownum = 3; colnum = 4; axislabelsize = 25; xlabel_loc = (0.5, 0.07); ylabel_loc = (0.07, 0.5)
    elif model_type in ['inversionsNEE-prior', 'inversionsNEE-prior_only_seasonal']:
        model_names = ['CAMS', 'CMS-Flux', 'CTE', 'MIROC', 'NISMON-CO2'] # remove inversions without providing priors
    elif model_type in ['UpscaledEC', 'UpscaledEC_only_seasonal', 'UpscaledEC_GPP']:
        model_names = ['X-BASE', 'ABCflux']
        figsize = (12, 6); rownum = 1; colnum = 2; axislabelsize = 20; xlabel_loc = (0.5, 0.03); ylabel_loc = (0.04, 0.5)
    elif model_type == 'UpscaledEC_Reco':
        model_names = ['X-BASE', 'ABCflux', 'ABCflux_upscaled']
        figsize = (20, 20); rownum = 4; colnum = 4; axislabelsize = 25; xlabel_loc = (0.5, 0.07); ylabel_loc = (0.07, 0.5)
    elif model_type == 'GPPobservations':
        model_names = ['GOSIF-GPP']
        figsize = (20, 20); rownum = 4; colnum = 4; axislabelsize = 25; xlabel_loc = (0.5, 0.07); ylabel_loc = (0.07, 0.5)
    elif model_type in ['reference', 'reference_only_seasonal']:
        model_names = ['APAR', 'LAI', 'FPAR', 'PAR', 'NDVI', 'EVI', 'GOME2_SIF']
        figsize = (20, 20); rownum = 4; colnum = 4; axislabelsize = 25; xlabel_loc = (0.5, 0.07); ylabel_loc = (0.07, 0.5)
    elif model_type == 'regression':
        model_names = ['CRU', 'constant', 'LC']
        figsize = (20, 20); rownum = 4; colnum = 4; axislabelsize = 25; xlabel_loc = (0.5, 0.07); ylabel_loc = (0.07, 0.5)

    if model_type in model_types_single:
        fig = plt.figure(figsize=figsize)

    for (model_id, model_name) in enumerate(model_names):

        print(model_name)

        # specify covariates if they are not constant + model_name
        if model_name == 'CRU':
            variable_names = ['dswrf', 'pre', 'spfh', 'tmp'] + ['constant']
        elif model_name == 'LC':
            variable_names = ['constant_forest', 'constant_shrub', 'constant_tundra', 'constant_others'] # do no need to include "constant" - adding "constant" results in the same statistics
        else:
            variable_names = ['constant'] # for constant, reference, and TRENDYv11 or inversionsNEE or UpscaledEC (transported NEE or variables will be added later)

        for year in [2012, 2013, 2014, 2017]: #2012, 2013, 2014, 2017

            start_month, end_month, campaign_name = get_campaign_info(year)
            month_num = end_month - start_month + 1

            # read atmospheric observations
            df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
            df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

            # filters for airborne observations
            mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
                (df_influence['ABoVE_influence_fraction'] > 0.5) &
                (df_influence['ocean_influence_fraction'] < 0.3) &
                (df_airborne['CO2_change'] < 30) &
                (df_airborne['CO_change'] < 40))[0]

            # # land cover filtering 1: select observations with footprint sensitivity of certain land cover > 50%
            # if lcname == 'forest':
            #     mask_id_lc = np.where(df_influence['forest_influence'] / df_influence['total_influence'] > 0.5)[0].tolist()
            #     mask_id = [i for i in mask_id if i in mask_id_lc]
            # elif lcname == 'shrub':
            #     mask_id_lc = np.where(df_influence['shrub_influence'] / df_influence['total_influence'] > 0.5)[0].tolist()
            #     mask_id = [i for i in mask_id if i in mask_id_lc]   
            # elif lcname == 'tundra':
            #     mask_id_lc = np.where(df_influence['tundra_influence'] / df_influence['total_influence'] > 0.5)[0].tolist()
            #     mask_id = [i for i in mask_id if i in mask_id_lc]     

            # land cover filtering 2: select observations with largest footprint sensitivity to certain land cover
            if lcname == 'forest':
                mask_id_lc = np.where((df_influence['forest_influence'] > df_influence['shrub_influence']) & 
                                      (df_influence['forest_influence'] > df_influence['tundra_influence']) & 
                                      (df_influence['forest_influence'] > df_influence['total_influence'] - df_influence['forest_influence'] - df_influence['shrub_influence'] - df_influence['tundra_influence'])
                                      )[0].tolist()
                mask_id = [i for i in mask_id if i in mask_id_lc]
            elif lcname == 'shrub':
                mask_id_lc = np.where((df_influence['shrub_influence'] > df_influence['forest_influence']) & 
                                      (df_influence['shrub_influence'] > df_influence['tundra_influence']) & 
                                      (df_influence['shrub_influence'] > df_influence['total_influence'] - df_influence['forest_influence'] - df_influence['shrub_influence'] - df_influence['tundra_influence'])
                                      )[0].tolist()
                mask_id = [i for i in mask_id if i in mask_id_lc]   
            elif lcname == 'tundra':
                mask_id_lc = np.where((df_influence['tundra_influence'] > df_influence['forest_influence']) & 
                                      (df_influence['tundra_influence'] > df_influence['shrub_influence']) & 
                                      (df_influence['tundra_influence'] > df_influence['total_influence'] - df_influence['forest_influence'] - df_influence['shrub_influence'] - df_influence['tundra_influence'])
                                      )[0].tolist()
                mask_id = [i for i in mask_id if i in mask_id_lc]     


            # influence from fossil and fire emissions
            df_fossil = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fossil.csv')
            df_fire = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fire.csv')

            # derive biogenic CO2 drawdown/enhancement from fossil and fire emissions
            y0 = df_airborne['CO2_change'] - df_fossil['gridfed2024'] - df_fire['gfed4.1']
            y_year = y0.loc[mask_id]


            # aggregate transported regression covariates from monthly to yearly
            for month in np.arange(start_month, end_month+1):
                
                for (variable_id, variable_name) in enumerate(variable_names):
                    
                    variable0 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/regression_covariates/{variable_name}_{year}_{month}.csv')
                    variable_month = variable0.loc[mask_id]

                    if variable_id == 0:
                        X_month = variable_month
                    else:
                        X_month = pd.concat((X_month, variable_month), axis=1)

                if month == start_month:
                    X_year = X_month
                else:
                    X_year += X_month
            

            '''add transported carbon fluxes or variables'''
            if model_type in model_types_single:
                
                if model_type in ['inversionsNEE', 'inversionsNEE-prior']: # account for fire emissions
                    if model_type == 'inversionsNEE':
                        df_model = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_inversions.csv')
                    elif model_type == 'inversionsNEE-prior':
                        df_model = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_inversions-prior.csv')
                    X_year_NEE = df_model[f"{model_name}"].loc[mask_id] - df_fire['gfed4.1'].loc[mask_id]
                    X_year = pd.concat((X_year_NEE, X_year), axis=1)
                else:
                    df_model = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_{model_type}.csv')
                    X_year_NEE = df_model[f"{model_name}"].loc[mask_id]
                    X_year = pd.concat((X_year_NEE, X_year), axis=1)

            if year == 2012:
                X = X_year
                y = y_year
            else:
                X = pd.concat((X, X_year), axis=0)
                y = pd.concat((y, y_year), axis=0)


        '''evaluate consistency between original surface fields and atmospheric observations, report cor'''
        if model_type in model_types_single:
            pearson_res = pearsonr(y, X.iloc[:,0])
            cor, _ = pearson_res
            cor_CI_low, cor_CI_high = pearson_res.confidence_interval(confidence_level=0.95)
            
            # can also use bootstrap but take much longer time
            # rng = np.random.default_rng()
            # method = stats.BootstrapMethod(method='BCa', random_state=rng)
            # pearson_res.confidence_interval(confidence_level=0.95, method=method)

            # mean bias and rmse
            mean_bias = np.mean(X.iloc[:,0] - y)
            rmse = np.sqrt(np.mean((X.iloc[:,0] - y)**2))

            # regression 1: HX ~ y
            # constant term in y, but no constant term in X
            # to directly evaluate the consistency between modeled and observed CO2 enhancement
            X_tmp = sm.add_constant(y) # observation as x-axis, modeled as y-axis
            model = sm.OLS(X.iloc[:,0], X_tmp)
            results1 = model.fit()
            slope = results1.params.iloc[1]
            intercept = results1.params.iloc[0]


            # Make scatterplot between observed and modeled CO2 enhancement
            plt.subplot(rownum,colnum,model_id+1,aspect='equal')
            # plt.scatter(y, X.iloc[:,0], s=60, alpha=0.3, edgecolors="none", color='tab:blue')
            
            # density scatter plot
            xy = np.vstack([y, X.iloc[:,0]])
            z = stats.gaussian_kde(xy)(xy)
            idx = z.argsort()
            x_sorted, y_sorted, z_sorted = y.iloc[idx], X.iloc[:,0].iloc[idx], z[idx]
            plt.scatter(x_sorted, y_sorted, c=z_sorted, s=60, alpha=0.3, edgecolors="none", cmap='viridis')

            x_line = np.arange(-50,50,0.1)
            plt.plot(x_line, x_line, color='black', linestyle='dashed') # Plot 1:1 line
            y_line = x_line * slope + intercept
            plt.plot(x_line, y_line, color='red', linestyle='dashed') # regression line
            # plt.annotate(r"$Cor$ = {:.2f}".format(cor), (45, -32), fontsize=20, ha='right', va='bottom')
            # plt.annotate('Bias = {:.2f}'.format(mean_bias), (45, -40), fontsize=20, ha='right', va='bottom')
            # plt.annotate('y={:.2f}x{:+.2f}'.format(slope, intercept), (45, -48), fontsize=20, ha='right', va='bottom')

            plt.xlim(-50, 50)
            plt.ylim(-50, 50)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            # plt.xlabel('Observed CO2 change (removing fossil and fire emissions)', fontsize=20)
            # plt.ylabel('Modeled CO2 change due to NEE', fontsize=15)

            if model_name == 'MIROC':
                model_name_label = 'MIROC-ACTM'
            else:
                model_name_label = model_name
            plt.text(-48, 48, f"({chr(97 + model_id)}) {model_name_label}", fontsize=25, va='top', ha='left')

            fitting_df_unscaled0 = pd.DataFrame([[model_name, cor, cor_CI_low, cor_CI_high, slope, intercept, mean_bias, rmse]],
                                                columns=['model_name', 'cor', 'cor_CI_low', 'cor_CI_high', 'slope', 'intercept', 'mean_bias', 'rmse']) 

            if model_id == 0:
                fitting_df_unscaled = fitting_df_unscaled0
            else:
                fitting_df_unscaled = pd.concat((fitting_df_unscaled, fitting_df_unscaled0))
            fitting_df_unscaled.to_csv(f"{dir0}evaluation_stat_unscaled_{model_type}{lc_filestr}_gridfed.csv", encoding='utf-8', index=False)


        '''evaluate consistency between scaled surface fields (with regression) and atmospheric observations'''
        # regression 2: y ~ HX
        # constant term in X, but no constant term in HX (eq. 2)
        # to properly scale remote sensing or CRU variables to NEE
        print(len(y))
        model = sm.OLS(y,X)
        results2 = model.fit()
        # results2.save(f"{dir1}{model_type}_{model_name}{lc_filestr}_gridfed.pickle")
        # # to load ols model back
        # results2 = OLSResults.load(f"{dir1}{model_type}_{model_name}{lc_filestr}_gridfed.pickle")
        # results2.params
        # results2.summary() # this is exactly the same as the results from fit_linear_regression_fullH.py
        # # export results to txt
        # f = f"{dir1}{model_type}_{model_name}{lc_filestr}_gridfed.txt"
        # with open(f, 'w') as fh:
        #     fh.write(results2.summary().as_text())

        # calculate correlation between z and H X beta
        y_hat = results2.fittedvalues
        pearson_res = pearsonr(y, y_hat)
        cor1, _ = pearson_res
        cor1_CI_low, cor1_CI_high = pearson_res.confidence_interval(confidence_level=0.95)
        r2_1 = 1 - np.sum((y-y_hat)**2)/np.sum((y-np.mean(y))**2) # explained variability

        # regression 3: y ~ y_hat (i.e., fitted NEE, y_hat = H X beta)
        # constant term in y_hat
        # to confirm the calculation of cor1
        X2 = sm.add_constant(y_hat)
        model = sm.OLS(y,X2)
        results3 = model.fit()
        results3.params
        r2_2 = results3.rsquared # cor1**2

        fitting_df0 = pd.DataFrame([[model_name, cor1, cor1_CI_low, cor1_CI_high, r2_1, r2_2]], 
                                columns=['model_name', 'cor', 'cor_CI_low', 'cor_CI_high', 'r2_1', 'r2_2'])
        
        if model_id == 0:
            fitting_df = fitting_df0
        else:
            fitting_df = pd.concat((fitting_df, fitting_df0))

    if model_type in model_types_single:
        fig.text(xlabel_loc[0], xlabel_loc[1], 'Observed CO$_2$ enhancement (ppm)', fontsize=axislabelsize, ha='center')
        fig.text(ylabel_loc[0], ylabel_loc[1], 'Modeled CO$_2$ enhancement (ppm)', fontsize=axislabelsize, va='center', rotation='vertical')
        plt.savefig(f"{dir0}evaluation_stat_{model_type}{lc_filestr}_gridfed_scatterplot.png", dpi=100, bbox_inches='tight')
        
        plt.show()
        
    fitting_df.to_csv(f"{dir0}evaluation_stat_scaled_{model_type}{lc_filestr}_gridfed.csv", encoding='utf-8', index=False)

    