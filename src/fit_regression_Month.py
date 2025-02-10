'''
fit a linear regression with Month as the only covariate
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
model_type = 'regression'
model_name = 'Month'

for year in [2012, 2013, 2014, 2017]: #, 2013, 2014, 2017

    start_month, end_month, campaign_name = get_campaign_info(year)
    month_num = end_month - start_month + 1

    # read observations
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

    # land cover filtering 2: select observations with largest footprint sensitivity of certain land cover
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

    # derive CO2 drawdown/enhancement from fossil and fire emissions
    y0 = df_airborne['CO2_change'].values - df_fossil_fire['fossil_CO2_change'] - df_fossil_fire['fire_CO2_change']
    y_year = y0.loc[mask_id]

    for month in np.arange(4, 12):

        filename = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/regression_covariates/constant_{year}_{month}.csv'
        if os.path.exists(filename):
            constant0 = pd.read_csv(filename)
        else:
            constant0 = pd.DataFrame(np.zeros(y0.shape), columns=['constant'])

        constant_year = constant0.loc[mask_id]
        constant_year = constant_year.rename(columns = {'constant': f'constant_{month}'})

        if month == 4:
            X_year = constant_year
        else:
            X_year = pd.concat((X_year, constant_year), axis=1)

    if year == 2012:
        X = X_year
        y = y_year
    else:
        X = pd.concat((X, X_year), axis=0)
        y = pd.concat((y, y_year), axis=0)


'''evaluate consistency between scaled surface fields (with regression) and atmospheric observations'''
# regression 2: y ~ HX
# constant term in X, but no constant term in HX (eq. 2)
# to properly scale remote sensing or CRU variables to NEE
model = sm.OLS(y,X)
results2 = model.fit()
results2.save(f"{dir1}{model_type}_{model_name}{lc_filestr}.pickle")
# to load ols model back
results2 = OLSResults.load(f"{dir1}{model_type}_{model_name}{lc_filestr}.pickle")
results2.params
results2.summary()
# export results to txt
f = f"{dir1}{model_type}_{model_name}{lc_filestr}.txt"
with open(f, 'w') as fh:
    fh.write(results2.summary().as_text())

# calculate correlation between z and H X beta
y_hat = results2.fittedvalues
pearson_res = pearsonr(y, y_hat)
cor1, _ = pearson_res
cor_CI_low, cor_CI_high = pearson_res.confidence_interval(confidence_level=0.95)
r2_1 = 1 - np.sum((y-y_hat)**2)/np.sum((y-np.mean(y))**2) # explained variability

# regression 3: y ~ y_hat (i.e., fitted NEE, y_hat = H X beta)
# constant term in y_hat
# to confirm the calculation of cor1
X2 = sm.add_constant(y_hat)
model = sm.OLS(y,X2)
results3 = model.fit()
results3.params
r2_2 = results3.rsquared

fitting_df = pd.DataFrame([[model_name, cor1, cor_CI_low, cor_CI_high, r2_1, r2_2]], 
                        columns=['model_name', 'cor', 'cor_CI_low', 'cor_CI_high', 'r2_1', 'r2_2'])
fitting_df.to_csv(f"{dir0}evaluation_stat_{model_name}{lc_filestr}.csv", encoding='utf-8', index=False)


#########################################################################
# plot seasonal vairations (i.e., fitted beta)
# standardize with maximum
def scale_maximum (vec):
    return vec / np.max(abs(vec))

# plot month-specific parameters
fig, ax = plt.subplots(figsize=(4,3))
scaling_factor = np.max(abs(results2.params))
# plt.plot(np.arange(4,12), results.params / scaling_factor, linestyle='-',color='black')
plt.errorbar(np.arange(4,12), results2.params / scaling_factor, yerr = results2.bse / scaling_factor,color='black', ecolor='red', elinewidth=3, capsize=5)
plt.xlim(4,11)
plt.ylim(-1.2,1)
ax.set_xticks(np.arange(4,12))
plt.xticks(fontsize=15)
plt.yticks(fontsize=12)
ax.tick_params(axis ='x', length = 7, direction ='in')
plt.xlabel('Month', fontsize=15)
plt.ylabel(f'Rescaled NEE from "Month" ' + '\n($\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=15)
plt.title(lcname, fontsize=20)


