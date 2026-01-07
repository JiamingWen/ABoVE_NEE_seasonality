'''plot a scatterplot between observed and modeled CO2 enhancement, as an example to show the evaluation metrics'''


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

model_type = 'TRENDYv11'
model_name = 'LPX-Bern' # ISBA-CTRIP SDGVM LPX-Bern

# whether to filter observations based on land covers they are most sensitive to
lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname


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
fig, ax = plt.subplots(figsize=(5,5))

# density scatter plot
xy = np.vstack([y, x])
z = stats.gaussian_kde(xy)(xy)
idx = z.argsort()
x_sorted, y_sorted, z_sorted = y.iloc[idx], x.iloc[idx], z[idx]
sc = plt.scatter(x_sorted, y_sorted, c=z_sorted, s=60, edgecolors="none", cmap='viridis')

x_line = np.arange(-70,70,0.1)
plt.plot(x_line, x_line, color='black', linestyle='dashed') # Plot 1:1 line

# plt.annotate(r"$Cor$ = {:.2f}".format(cor), (45, -32), fontsize=20, ha='right', va='bottom')
# plt.annotate('Bias = {:.2f}'.format(mean_bias), (45, -40), fontsize=20, ha='right', va='bottom')
# plt.annotate('y={:.2f}x{:+.2f}'.format(slope, intercept), (45, -48), fontsize=20, ha='right', va='bottom')

plt.xlim(-70, 70)
plt.ylim(-70, 70)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# plt.xlabel('Observed CO2 change (removing fossil and fire emissions)', fontsize=20)
# plt.ylabel('Modeled CO2 change due to NEE', fontsize=15)

ax.set_xlabel(r'$z_{\mathrm{obs}}$ (ppm)', fontsize=25)
ax.set_ylabel(r'$z_{\mathrm{model}}$ (ppm)', fontsize=25)

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.3, 0.03, 0.4])  # [left, bottom, width, height]
# fig.colorbar(sc, cax=cbar_ax)
# cbar_ax.tick_params(labelsize=18)
# cbar_ax.set_ylabel("Scatter density", fontsize=22)
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar_ax.tick_params(labelsize=18)
cbar.set_ticks([z.min(), z.max()])
cbar.set_ticklabels(['Low', 'High'])
cbar_ax.set_ylabel("Scatter density", fontsize=22)

plt.show()