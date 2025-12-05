'''evaluate certain models with diurnal cycle outputs
CT-NOAA: 3-hourly, daily, monthly
CTE: 3-hourly, daily, monthly
X-BASE: hourly, daily, monthly
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLSResults

os.chdir('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info

dir0 = f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/"
dir1 = f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/regression/"

# whether to filter observations based on land covers they are most sensitive to
lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

model_types = ['CT-NOAA', 'CTE', 'X-BASE']

for model_type in model_types:
    if model_type in ['CT-NOAA', 'CTE']:
        model_names = [model_type+'-3hourly', model_type+'-daily', model_type+'-monthly']
    elif model_type == 'X-BASE':
        model_names = [model_type+'-monthly_diurnal', model_type+'-daily', model_type+'-monthly']

    figsize = (20, 7); rownum = 1; colnum = 3; axislabelsize = 25; xlabel_loc = (0.5, 0.07); ylabel_loc = (0.07, 0.5)
    fig = plt.figure(figsize=figsize)

    for (model_id, model_name) in enumerate(model_names):

        for year in [2012, 2013, 2014, 2017]: #2012, 2013, 2014, 2017

            start_month, end_month, campaign_name = get_campaign_info(year)
            month_num = end_month - start_month + 1

            # read atmospheric observations
            df_airborne = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
            df_influence = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

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
            df_fossil = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fossil.csv')
            df_fire = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fire.csv')

            # derive biogenic CO2 drawdown/enhancement from fossil and fire emissions
            y0 = df_airborne['CO2_change'] - df_fossil['odiac2022'] - df_fire['gfed4.1']
            y_year = y0.loc[mask_id]


            # aggregate transported regression covariates from monthly to yearly
            for month in np.arange(start_month, end_month+1):
                
                variable_names = ['constant']

                for (variable_id, variable_name) in enumerate(variable_names):
                    
                    variable0 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/regression_covariates/{variable_name}_{year}_{month}.csv')
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
            if model_type in ['CT-NOAA', 'CTE']:
                df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_{model_name}.csv')
                X_year_NEE = df_model[f"{model_type}"].loc[mask_id] - df_fire['gfed4.1'].loc[mask_id]
                X_year = pd.concat((X_year_NEE, X_year), axis=1)
            elif model_type == 'X-BASE':
                df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_{model_name}.csv')
                X_year_NEE = df_model[f"{model_type}"].loc[mask_id]
                X_year = pd.concat((X_year_NEE, X_year), axis=1)               
            else:
                print('model type not recognized')
                
            if year == 2012:
                X = X_year
                y = y_year
            else:
                X = pd.concat((X, X_year), axis=0)
                y = pd.concat((y, y_year), axis=0)


        '''evaluate consistency between original surface fields and atmospheric observations, report cor'''
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
        # specify z range - derived from plot_obs_model_co2_enhancement.py
        zmin = 6.48404771703154e-06
        zmax = 0.05048420795712829 * 0.75 # apply a factor to shrink the range
        sc = plt.scatter(x_sorted, y_sorted, c=z_sorted, s=60, edgecolors="none", cmap='viridis', vmin=zmin, vmax=zmax)

        x_line = np.arange(-50,50,0.1)
        plt.plot(x_line, x_line, color='black', linestyle='dashed') # Plot 1:1 line
        y_line = x_line * slope + intercept
        plt.plot(x_line, y_line, color='red', linestyle='dashed') # regression line
        plt.annotate(r"$Cor$ = {:.2f}".format(cor), (45, -32), fontsize=20, ha='right', va='bottom')
        plt.annotate('Bias = {:.2f}'.format(mean_bias), (45, -40), fontsize=20, ha='right', va='bottom')
        plt.annotate('y={:.2f}x{:+.2f}'.format(slope, intercept), (45, -48), fontsize=20, ha='right', va='bottom')

        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        # plt.xlabel('Observed CO2 change (removing fossil and fire emissions)', fontsize=20)
        # plt.ylabel('Modeled CO2 change due to NEE', fontsize=15)

        model_name_label = model_name
        plt.text(-48, 48, f"({chr(97 + model_id)}) {model_name_label}", fontsize=20, va='top', ha='left')

        fitting_df_unscaled0 = pd.DataFrame([[model_name, cor, cor_CI_low, cor_CI_high, slope, intercept, mean_bias, rmse]],
                                            columns=['model_name', 'cor', 'cor_CI_low', 'cor_CI_high', 'slope', 'intercept', 'mean_bias', 'rmse']) 

        if model_id == 0:
            fitting_df_unscaled = fitting_df_unscaled0
        else:
            fitting_df_unscaled = pd.concat((fitting_df_unscaled, fitting_df_unscaled0))
        fitting_df_unscaled.to_csv(f"{dir0}evaluation_stat_unscaled_{model_type}{lc_filestr}_highres.csv", encoding='utf-8', index=False)


        '''evaluate consistency between scaled surface fields (with regression) and atmospheric observations'''
        # regression 2: y ~ HX
        # constant term in X, but no constant term in HX (eq. 2)
        # to properly scale remote sensing or CRU variables to NEE
        print(len(y))
        model = sm.OLS(y,X)
        results2 = model.fit()
        results2.save(f"{dir1}{model_type}_{model_name}{lc_filestr}_highres.pickle")
        # to load ols model back
        results2 = OLSResults.load(f"{dir1}{model_type}_{model_name}{lc_filestr}_highres.pickle")
        results2.params
        results2.summary() # this is exactly the same as the results from fit_linear_regression_fullH.py
        # export results to txt
        f = f"{dir1}{model_type}_{model_name}{lc_filestr}_highres.txt"
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

    fig.text(xlabel_loc[0], xlabel_loc[1], 'Observed CO$_2$ enhancement (ppm)', fontsize=axislabelsize, ha='center')
    fig.text(ylabel_loc[0], ylabel_loc[1], 'Modeled CO$_2$ enhancement (ppm)', fontsize=axislabelsize, va='center', rotation='vertical')
    
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.75])  # [left, bottom, width, height]
    fig.colorbar(sc, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=18)
    cbar_ax.set_ylabel("Scatter density", fontsize=22)

    plt.savefig(f"{dir0}evaluation_stat_{model_type}{lc_filestr}_highres_scatterplot.png", dpi=100, bbox_inches='tight')
    
    if model_type in ['CT-NOAA', 'CTE', 'X-BASE']:
        plt.savefig(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitiviity_test_high_res_nee/evaluation_stat_{model_type}{lc_filestr}_highres_scatterplot.png", dpi=100, bbox_inches='tight')
        plt.savefig(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitiviity_test_high_res_nee/evaluation_stat_{model_type}{lc_filestr}_highres_scatterplot.pdf", dpi=100, bbox_inches='tight')
    plt.show()
        
    fitting_df.to_csv(f"{dir0}evaluation_stat_scaled_{model_type}{lc_filestr}_highres.csv", encoding='utf-8', index=False)

