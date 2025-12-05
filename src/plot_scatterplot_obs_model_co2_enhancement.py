'''plot scatterplots between observed and modeled CO2 enhancement'''

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

# whether to filter observations based on land covers they are most sensitive to
lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

model_types = ['TRENDYv11', 'inversionsNEE', 'UpscaledEC'] # 


# '''determine the range of color bar'''
# all_z = []
# for model_type in model_types:
#     if model_type == 'TRENDYv11':
#         model_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
#         figsize = (20, 20); rownum = 4; colnum = 4; axislabelsize = 25; xlabel_loc = (0.5, 0.07); ylabel_loc = (0.07, 0.5)
#     elif model_type == 'inversionsNEE':
#         model_names = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE'] # remove the five satellite-based inversions
#         figsize = (20, 14); rownum = 3; colnum = 4; axislabelsize = 25; xlabel_loc = (0.5, 0.07); ylabel_loc = (0.07, 0.5)
#     elif model_type == 'UpscaledEC':
#         model_names = ['X-BASE', 'ABCflux']
#         figsize = (12, 6); rownum = 1; colnum = 2; axislabelsize = 20; xlabel_loc = (0.5, 0.03); ylabel_loc = (0.04, 0.5)

#     fig = plt.figure(figsize=figsize)

#     for (model_id, model_name) in enumerate(model_names):

#         print(model_name)

#         for year in [2012, 2013, 2014, 2017]: #2012, 2013, 2014, 2017

#             start_month, end_month, campaign_name = get_campaign_info(year)
#             month_num = end_month - start_month + 1

#             # read atmospheric observations
#             df_airborne = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
#             df_influence = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

#             # filters for airborne observations
#             mask_id = np.where((df_airborne['background_CO2_std'].notna()) &
#                 (df_influence['ABoVE_influence_fraction'] > 0.5) &
#                 (df_influence['ocean_influence_fraction'] < 0.3) &
#                 (df_airborne['CO2_change'] < 30) &
#                 (df_airborne['CO_change'] < 40))[0]

#             # # land cover filtering 1: select observations with footprint sensitivity of certain land cover > 50%
#             # if lcname == 'forest':
#             #     mask_id_lc = np.where(df_influence['forest_influence'] / df_influence['total_influence'] > 0.5)[0].tolist()
#             #     mask_id = [i for i in mask_id if i in mask_id_lc]
#             # elif lcname == 'shrub':
#             #     mask_id_lc = np.where(df_influence['shrub_influence'] / df_influence['total_influence'] > 0.5)[0].tolist()
#             #     mask_id = [i for i in mask_id if i in mask_id_lc]   
#             # elif lcname == 'tundra':
#             #     mask_id_lc = np.where(df_influence['tundra_influence'] / df_influence['total_influence'] > 0.5)[0].tolist()
#             #     mask_id = [i for i in mask_id if i in mask_id_lc]     

#             # land cover filtering 2: select observations with largest footprint sensitivity to certain land cover
#             if lcname == 'forest':
#                 mask_id_lc = np.where((df_influence['forest_influence'] > df_influence['shrub_influence']) & 
#                                         (df_influence['forest_influence'] > df_influence['tundra_influence']) & 
#                                         (df_influence['forest_influence'] > df_influence['total_influence'] - df_influence['forest_influence'] - df_influence['shrub_influence'] - df_influence['tundra_influence'])
#                                         )[0].tolist()
#                 mask_id = [i for i in mask_id if i in mask_id_lc]
#             elif lcname == 'shrub':
#                 mask_id_lc = np.where((df_influence['shrub_influence'] > df_influence['forest_influence']) & 
#                                         (df_influence['shrub_influence'] > df_influence['tundra_influence']) & 
#                                         (df_influence['shrub_influence'] > df_influence['total_influence'] - df_influence['forest_influence'] - df_influence['shrub_influence'] - df_influence['tundra_influence'])
#                                         )[0].tolist()
#                 mask_id = [i for i in mask_id if i in mask_id_lc]   
#             elif lcname == 'tundra':
#                 mask_id_lc = np.where((df_influence['tundra_influence'] > df_influence['forest_influence']) & 
#                                         (df_influence['tundra_influence'] > df_influence['shrub_influence']) & 
#                                         (df_influence['tundra_influence'] > df_influence['total_influence'] - df_influence['forest_influence'] - df_influence['shrub_influence'] - df_influence['tundra_influence'])
#                                         )[0].tolist()
#                 mask_id = [i for i in mask_id if i in mask_id_lc]     


#             # influence from fossil and fire emissions
#             df_fossil = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fossil.csv')
#             df_fire = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fire.csv')

#             # derive biogenic CO2 drawdown/enhancement from fossil and fire emissions
#             y0 = df_airborne['CO2_change'].values - df_fossil['odiac2022'] - df_fire['gfed4.1']
#             y_year = y0.loc[mask_id]

                
#             if model_type in ['inversionsNEE', 'inversionsNEE-prior']: # account for fire emissions
#                 if model_type == 'inversionsNEE':
#                     df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_inversions.csv')
#                 elif model_type == 'inversionsNEE-prior':
#                     df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_inversions-prior.csv')
#                 x_year = df_model[f"{model_name}"].loc[mask_id] - df_fire['gfed4.1'].loc[mask_id]
#             else:
#                 df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_{model_type}.csv')
#                 x_year = df_model[f"{model_name}"].loc[mask_id]

#             if year == 2012:
#                 x = x_year
#                 y = y_year
#             else:
#                 x = pd.concat((x, x_year), axis=0)
#                 y = pd.concat((y, y_year), axis=0)

#         # statistics
#         pearson_res = pearsonr(y, x)
#         cor, _ = pearson_res
#         cor_CI_low, cor_CI_high = pearson_res.confidence_interval(confidence_level=0.95)
        
#         # can also use bootstrap but take much longer time
#         # rng = np.random.default_rng()
#         # method = stats.BootstrapMethod(method='BCa', random_state=rng)
#         # pearson_res.confidence_interval(confidence_level=0.95, method=method)

#         # mean bias and rmse
#         mean_bias = np.mean(x - y)
#         rmse = np.sqrt(np.mean((x - y)**2))

#         # regression 1: HX ~ y
#         # constant term in y, but no constant term in X
#         # to directly evaluate the consistency between modeled and observed CO2 enhancement
#         X_tmp = sm.add_constant(y) # observation as x-axis, modeled as y-axis
#         model = sm.OLS(x, X_tmp)
#         results1 = model.fit()
#         slope = results1.params.iloc[1]
#         intercept = results1.params.iloc[0]

#         # Make scatterplot between observed and modeled CO2 enhancement
#         plt.subplot(rownum,colnum,model_id+1,aspect='equal')
        
#         # density scatter plot
#         xy = np.vstack([y, x])
#         z = stats.gaussian_kde(xy)(xy)
#         print(np.min(z), np.max(z))
#         all_z.append(z)
#         idx = z.argsort()
#         x_sorted, y_sorted, z_sorted = y.iloc[idx], x.iloc[idx], z[idx]
#         plt.scatter(x_sorted, y_sorted, c=z_sorted, s=60, alpha=0.3, edgecolors="none", cmap='viridis')

#         x_line = np.arange(-50,50,0.1)
#         plt.plot(x_line, x_line, color='black', linestyle='dashed') # Plot 1:1 line
#         y_line = x_line * slope + intercept
#         plt.plot(x_line, y_line, color='red', linestyle='dashed') # regression line
#         # plt.annotate(r"$Cor$ = {:.2f}".format(cor), (45, -32), fontsize=20, ha='right', va='bottom')
#         # plt.annotate('Bias = {:.2f}'.format(mean_bias), (45, -40), fontsize=20, ha='right', va='bottom')
#         # plt.annotate('y={:.2f}x{:+.2f}'.format(slope, intercept), (45, -48), fontsize=20, ha='right', va='bottom')

#         plt.xlim(-50, 50)
#         plt.ylim(-50, 50)
#         plt.xticks(fontsize=18)
#         plt.yticks(fontsize=18)
#         ax = plt.gca()
#         ax.set_aspect('equal', adjustable='box')
#         # plt.xlabel('Observed CO2 change (removing fossil and fire emissions)', fontsize=20)
#         # plt.ylabel('Modeled CO2 change due to NEE', fontsize=15)

#         if model_name == 'MIROC':
#             model_name_label = 'MIROC-ACTM'
#         else:
#             model_name_label = model_name
#         plt.text(-48, 48, f"({chr(97 + model_id)}) {model_name_label}", fontsize=25, va='top', ha='left')

#     fig.text(xlabel_loc[0], xlabel_loc[1], 'Observed CO$_2$ enhancement (ppm)', fontsize=axislabelsize, ha='center')
#     fig.text(ylabel_loc[0], ylabel_loc[1], 'Modeled CO$_2$ enhancement (ppm)', fontsize=axislabelsize, va='center', rotation='vertical')

#     # plt.savefig(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/evaluation_stat_{model_type}{lc_filestr}_scatterplot.png", dpi=100, bbox_inches='tight')
#     # plt.savefig(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/evaluation_stat_{model_type}{lc_filestr}_scatterplot.pdf", dpi=100, bbox_inches='tight')
    
#     plt.show()

# zmin, zmax = np.min(all_z), np.max(all_z)
# print(zmin, zmax) #6.48404771703154e-06 0.05048420795712829

zmin = 6.48404771703154e-06
zmax = 0.05048420795712829 * 0.75 # apply a factor to shrink the range

'''real scatterplots'''
for model_type in model_types:
    if model_type == 'TRENDYv11':
        model_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
        figsize = (20, 20); rownum = 4; colnum = 4; axislabelsize = 25; xlabel_loc = (0.5, 0.07); ylabel_loc = (0.07, 0.5)
    elif model_type == 'inversionsNEE':
        model_names = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE'] # remove the five satellite-based inversions
        figsize = (20, 14); rownum = 3; colnum = 4; axislabelsize = 25; xlabel_loc = (0.5, 0.07); ylabel_loc = (0.07, 0.5)
    elif model_type == 'UpscaledEC':
        model_names = ['X-BASE', 'ABCflux']
        figsize = (12, 6); rownum = 1; colnum = 2; axislabelsize = 20; xlabel_loc = (0.5, 0.03); ylabel_loc = (0.04, 0.5)

    fig = plt.figure(figsize=figsize)
    sc = None

    for (model_id, model_name) in enumerate(model_names):

        print(model_name)

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
            y0 = df_airborne['CO2_change'].values - df_fossil['odiac2022'] - df_fire['gfed4.1']
            y_year = y0.loc[mask_id]

                
            if model_type in ['inversionsNEE', 'inversionsNEE-prior']: # account for fire emissions
                if model_type == 'inversionsNEE':
                    df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_inversions.csv')
                elif model_type == 'inversionsNEE-prior':
                    df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_inversions-prior.csv')
                x_year = df_model[f"{model_name}"].loc[mask_id] - df_fire['gfed4.1'].loc[mask_id]
            else:
                df_model = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_{model_type}.csv')
                x_year = df_model[f"{model_name}"].loc[mask_id]

            if year == 2012:
                x = x_year
                y = y_year
            else:
                x = pd.concat((x, x_year), axis=0)
                y = pd.concat((y, y_year), axis=0)

        # statistics
        pearson_res = pearsonr(y, x)
        cor, _ = pearson_res
        cor_CI_low, cor_CI_high = pearson_res.confidence_interval(confidence_level=0.95)
        
        # can also use bootstrap but take much longer time
        # rng = np.random.default_rng()
        # method = stats.BootstrapMethod(method='BCa', random_state=rng)
        # pearson_res.confidence_interval(confidence_level=0.95, method=method)

        # mean bias and rmse
        mean_bias = np.mean(x - y)
        rmse = np.sqrt(np.mean((x - y)**2))

        # regression 1: HX ~ y
        # constant term in y, but no constant term in X
        # to directly evaluate the consistency between modeled and observed CO2 enhancement
        X_tmp = sm.add_constant(y) # observation as x-axis, modeled as y-axis
        model = sm.OLS(x, X_tmp)
        results1 = model.fit()
        slope = results1.params.iloc[1]
        intercept = results1.params.iloc[0]

        # Make scatterplot between observed and modeled CO2 enhancement
        plt.subplot(rownum,colnum,model_id+1,aspect='equal')
        
        # density scatter plot
        xy = np.vstack([y, x])
        z = stats.gaussian_kde(xy)(xy)
        idx = z.argsort()
        x_sorted, y_sorted, z_sorted = y.iloc[idx], x.iloc[idx], z[idx]
        sc = plt.scatter(x_sorted, y_sorted, c=z_sorted, s=60, edgecolors="none", cmap='viridis', vmin=zmin, vmax=zmax)

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

    fig.text(xlabel_loc[0], xlabel_loc[1], 'Observed CO$_2$ enhancement (ppm)', fontsize=axislabelsize, ha='center')
    fig.text(ylabel_loc[0], ylabel_loc[1], 'Modeled CO$_2$ enhancement (ppm)', fontsize=axislabelsize, va='center', rotation='vertical')

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.75])  # [left, bottom, width, height]
    fig.colorbar(sc, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=18)
    cbar_ax.set_ylabel("Scatter density", fontsize=22)

    plt.savefig(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/evaluation_stat_{model_type}{lc_filestr}_scatterplot.png", dpi=100, bbox_inches='tight')
    plt.savefig(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/evaluation_stat_{model_type}{lc_filestr}_scatterplot.pdf", dpi=100, bbox_inches='tight')
    
    plt.show()
