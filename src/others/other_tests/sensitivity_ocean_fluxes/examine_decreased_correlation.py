'''examine why correlation decreases when I remove the data filtering criterion for regional relevance'''


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

model_type = 'TRENDYv11'
model_name = 'ISBA-CTRIP'

# whether to filter observations based on land covers they are most sensitive to
lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname


def plot_density_scatterplot (x, y, title):
    xy = np.vstack([y, x])
    z = stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = x[idx], y[idx], z[idx]
    plt.scatter(x_sorted, y_sorted, c=z_sorted, s=60, alpha=0.3, edgecolors="none", cmap='viridis')

    x_line = np.arange(-50,50,0.1)
    plt.plot(x_line, x_line, color='black', linestyle='dashed') # Plot 1:1 line

    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('Observed CO$_2$ enhancement (ppm)', fontsize=20)
    plt.ylabel('Modeled CO$_2$ enhancement (ppm)', fontsize=15)
    plt.title(title, fontsize=20)
    return 

for year in [2012, 2013, 2014, 2017]: #2012, 2013, 2014, 2017

    print(year)

    start_month, end_month, campaign_name = get_campaign_info(year)
    month_num = end_month - start_month + 1

    # read atmospheric observations
    df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
    df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')

    # filters for airborne observations
    mask_id_all = np.where((df_airborne['background_CO2_std'].notna()) &
        (df_influence['ABoVE_influence_fraction'] > 0.5) &
        (df_influence['ocean_influence_fraction'] < 0.3) &
        (df_airborne['CO2_change'] < 30) &
        (df_airborne['CO_change'] < 40))[0]

    mask_id_part = np.where((df_airborne['background_CO2_std'].notna()) &
        (df_airborne['CO2_change'] < 30) &
        (df_airborne['CO_change'] < 40))[0]

    # influence from fossil and fire emissions
    df_fossil = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fossil.csv')
    df_fire = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fire.csv')

    # derive biogenic CO2 drawdown/enhancement from fossil and fire emissions
    y0 = df_airborne['CO2_change'] - df_fossil['odiac2022'] - df_fire['gfed4.1']
    obs_year_all = y0.loc[mask_id_all].values
    obs_year_part = y0.loc[mask_id_part].values

    # modeled co2 enhancements
    df_model = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_{model_type}.csv')
    model_year_all = df_model[f"{model_name}"].loc[mask_id_all].values
    model_year_part = df_model[f"{model_name}"].loc[mask_id_part].values

    if year == 2012:
        model_all = model_year_all
        model_part = model_year_part
        obs_all = obs_year_all
        obs_part = obs_year_part
    else:
        model_all = np.concatenate((model_all, model_year_all), axis=0)
        model_part = np.concatenate((model_part, model_year_part), axis=0)
        obs_all = np.concatenate((obs_all, obs_year_all), axis=0)
        obs_part = np.concatenate((obs_part, obs_year_part), axis=0)

    # plot scatterplots for each year's data
    cor_year_all, _ = pearsonr(obs_year_all, model_year_all)
    cor_year_part, _ = pearsonr(obs_year_part, model_year_part)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1) # Panel 1: All data
    plot_density_scatterplot(obs_all, model_all, title=f"With Ocean Filter {year}\n(Cor: {cor_year_all:.2f}, n={len(obs_year_all)})")
    plt.subplot(1, 2, 2) # Panel 2: Part data
    plot_density_scatterplot(obs_part, model_part, title=f"Without Ocean Filter {year}\n(Cor: {cor_year_part:.2f}, n={len(obs_year_part)})")
    plt.tight_layout()
    plt.show()
    

'''plot scatterplots for all years' data'''
cor_all, _ = pearsonr(obs_all, model_all)
cor_part, _ = pearsonr(obs_part, model_part)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) # Panel 1: All data
plot_density_scatterplot(obs_all, model_all, title=f"With Ocean Filter\n(Cor: {cor_all:.2f}, n={len(obs_all)})")
plt.subplot(1, 2, 2) # Panel 2: Part data
plot_density_scatterplot(obs_part, model_part, title=f"Without Ocean Filter\n(Cor: {cor_part:.2f}, n={len(obs_part)})")
plt.tight_layout()
plt.show()