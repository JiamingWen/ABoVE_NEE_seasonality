'''
calculate statistics (e.g., Pearson correlation, slope, intercept) between modeled and observed CO2 enhancement
calculate for both original unscaled values and scaled values with regression
use single year's data
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLSResults

os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info

# create dir
dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/"
if not os.path.exists(dir0):
    os.makedirs(dir0)

dir1 = f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/regression/"
if not os.path.exists(dir1):
    os.makedirs(dir1)

# whether to filter observations based on land covers they are most sensitive to
lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

# model types
# all types
model_types = ['TRENDYv11', 'inversionsNEE', 'reference', 'regression', 'TRENDYv11GPP', 'TRENDYv11NPP', 'TRENDYv11Reco', 'NEEobservations', 'GPPobservations', 'TRENDYv11_only_seasonal', 'inversionsNEE_only_seasonal', 'NEEobservations_only_seasonal', 'reference_only_seasonal', 'inversionsNEE-prior', 'inversionsNEE-prior_only_seasonal'] #'TRENDYv11', 'inversionsNEE', 'reference', 'regression', 'TRENDYv11GPP', 'TRENDYv11NPP', 'TRENDYv11Reco', 'NEEobservations', 'GPPobservations', 'TRENDYv11_only_seasonal', 'inversionsNEE_only_seasonal', 'NEEobservations_only_seasonal', 'reference_only_seasonal', 'inversionsNEE-prior', 'inversionsNEE-prior_only_seasonal'
# only one variable (i.e., not regression like CRU or LC)
model_types_single = ['TRENDYv11', 'inversionsNEE', 'reference', 'TRENDYv11GPP', 'TRENDYv11NPP', 'TRENDYv11Reco', 'NEEobservations', 'GPPobservations', 'TRENDYv11_only_seasonal', 'inversionsNEE_only_seasonal', 'NEEobservations_only_seasonal', 'reference_only_seasonal', 'inversionsNEE-prior', 'inversionsNEE-prior_only_seasonal']

for year in [2012, 2013, 2014, 2017]: #2012, 2013, 2014, 2017

    start_month, end_month, campaign_name = get_campaign_info(year)
    month_num = end_month - start_month + 1

    # read atmospheric observations
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
    df_fossil_fire = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fossil_fire.csv')

    # derive biogenic CO2 drawdown/enhancement from fossil and fire emissions
    y0 = df_airborne['CO2_change'].values - df_fossil_fire['fossil_CO2_change'] - df_fossil_fire['fire_CO2_change']
    y_year = y0.loc[mask_id]    
    

    for model_type in model_types:
        if model_type in ['TRENDYv11', 'TRENDYv11GPP', 'TRENDYv11NPP', 'TRENDYv11Reco', 'TRENDYv11_only_seasonal']:
            model_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
        elif model_type in ['inversionsNEE','inversionsNEE_only_seasonal']:
            if year == 2017:
                model_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
            else:
                model_names = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE'] # remove the five satellite-based inversions
        elif model_type in ['inversionsNEE-prior']:
            if year == 2017:
                model_names = ['CAMS', 'CAMS-Satellite', 'CMS-Flux', 'CTE', 'GCASv2', 'GONGGA', 'MIROC', 'NISMON-CO2', 'THU']
            else:
                model_names = ['CAMS', 'CMS-Flux', 'CTE', 'MIROC', 'NISMON-CO2'] # remove inversions without providing priors
        elif model_type == 'inversionsNEE-prior_only_seasonal':
            model_names = ['CAMS', 'CMS-Flux', 'CTE', 'MIROC', 'NISMON-CO2']
        elif model_type in ['NEEobservations', 'NEEobservations_only_seasonal']:
            model_names = ['FluxCOM-X-NEE', 'ABCflux-NEE']
        elif model_type == 'GPPobservations':
            model_names = ['GOSIF-GPP', 'FluxCOM-X-GPP', 'ABCflux-GPP']
        elif model_type in ['reference', 'reference_only_seasonal']:
            model_names = ['APAR', 'LAI', 'FPAR', 'PAR', 'NDVI', 'EVI', 'GOME2_SIF']
        elif model_type == 'regression':
            model_names = ['CRU', 'constant', 'LC']

        if model_type in model_types_single:
            fig = plt.figure(figsize=(20, 5 * (len(model_names) // 4 + 1)))

        for (model_id, model_name) in enumerate(model_names):

            # specify covariates if they are not constant + model_name
            if model_name == 'CRU':
                variable_names = ['dswrf', 'pre', 'spfh', 'tmp'] + ['constant']
            elif model_name == 'LC':
                variable_names = ['constant_forest', 'constant_shrub', 'constant_tundra', 'constant_others'] # do no need to include "constant" - adding "constant" results in the same statistics
            else:
                variable_names = ['constant'] # for constant, reference, and TRENDYv11 or inversionsNEE or NEEobservations (transported NEE or variables will be added later)


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
                    X_year_NEE = df_model[f"{model_name}_CO2_change"].loc[mask_id] - df_fossil_fire['fire_CO2_change'].loc[mask_id]
                    X_year = pd.concat((X_year_NEE, X_year), axis=1)
                else:
                    df_model = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_{model_type}.csv')
                    X_year_NEE = df_model[f"{model_name}_CO2_change"].loc[mask_id]
                    X_year = pd.concat((X_year_NEE, X_year), axis=1)

            # use single year's data
            X = X_year
            y = y_year

            '''evaluate consistency between original surface fields and atmospheric observations, report cor'''
            if model_type in model_types_single:
                pearson_res = pearsonr(y, X.iloc[:,0])
                cor, _ = pearson_res
                cor_CI_low, cor_CI_high = pearson_res.confidence_interval(confidence_level=0.95)
                
                # can also use bootstrap but take much longer time
                # rng = np.random.default_rng()
                # method = stats.BootstrapMethod(method='BCa', random_state=rng)
                # pearson_res.confidence_interval(confidence_level=0.95, method=method)

                # regression 1: HX ~ y
                # constant term in y, but no constant term in X
                # to directly evaluate the consistency between modeled and observed CO2 enhancement
                X_tmp = sm.add_constant(y) # observation as x-axis, modeled as y-axis
                model = sm.OLS(X.iloc[:,0], X_tmp)
                results1 = model.fit()
                slope = results1.params.iloc[1]
                intercept = results1.params.iloc[0]


                # Make scatterplot between observed and modeled CO2 enhancement
                plt.subplot(np.ceil(len(model_names)/4).astype(int),4,model_id+1,aspect='equal')
                plt.scatter(y, X.iloc[:,0], s=60, alpha=0.7, edgecolors="k")
                x_line = np.arange(-30,30,0.1)
                plt.plot(x_line, x_line, color='black', linestyle='dashed') # Plot 1:1 line
                y_line = x_line * slope + intercept
                plt.plot(x_line, y_line, color='red', linestyle='dashed') # regression line
                plt.annotate(r"$Cor$ = {:.2f}".format(cor), (-28, 25), fontsize=20)
                plt.annotate('y={:.2f}x+{:.2f}'.format(slope, intercept), (-28, 21), fontsize=20)

                plt.xlim(-30, 30)
                plt.ylim(-30, 30)
                ax = plt.gca()
                ax.set_aspect('equal', adjustable='box')
                # plt.xlabel('Observed CO2 change (removing fossil and fire emissions)', fontsize=20)
                # plt.ylabel('Modeled CO2 change due to NEE', fontsize=15)
                plt.title(model_name, fontsize=25)

                fitting_df_unscaled0 = pd.DataFrame([[model_name, cor, cor_CI_low, cor_CI_high, slope, intercept]], 
                                        columns=['model_name', 'cor', 'cor_CI_low', 'cor_CI_high', 'slope', 'intercept'])
                if model_id == 0:
                    fitting_df_unscaled = fitting_df_unscaled0
                else:
                    fitting_df_unscaled = pd.concat((fitting_df_unscaled, fitting_df_unscaled0))
                fitting_df_unscaled.to_csv(f"{dir0}evaluation_stat_unscaled_{model_type}{lc_filestr}_{year}.csv", encoding='utf-8', index=False)


            '''evaluate consistency between scaled surface fields (with regression) and atmospheric observations'''
            # regression 2: y ~ HX
            # constant term in X, but no constant term in HX (eq. 2)
            # to properly scale remote sensing or CRU variables to NEE
            print(len(y))
            model = sm.OLS(y,X)
            results2 = model.fit()
            results2.save(f"{dir1}{model_type}_{model_name}{lc_filestr}_{year}.pickle")
            # to load ols model back
            results2 = OLSResults.load(f"{dir1}{model_type}_{model_name}{lc_filestr}_{year}.pickle")
            results2.params
            results2.summary() # this is exactly the same as the results from fit_linear_regression_fullH.py
            # export results to txt
            f = f"{dir1}{model_type}_{model_name}{lc_filestr}_{year}.txt"
            with open(f, 'w') as fh:
                fh.write(results2.summary().as_text())

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
            fig.text(0.5, 0.07, 'Observed CO2 enhancement (ppm)', fontsize=25, ha='center')
            fig.text(0.08, 0.5, 'Transported surface field', fontsize=25, va='center', rotation='vertical')
            plt.savefig(f"{dir0}evaluation_stat_{model_type}{lc_filestr}_{year}_scatterplot.png", dpi=100, bbox_inches='tight')
            plt.show()
            
        fitting_df.to_csv(f"{dir0}evaluation_stat_scaled_{model_type}{lc_filestr}_{year}.csv", encoding='utf-8', index=False)

    