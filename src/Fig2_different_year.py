'''
plot summary figure for model performance (correlation with observed CO2 enhancement)
evaluation separately for each year
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import get_campaign_info
from statsmodels.regression.linear_model import OLSResults

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

stat_var = 'cor'; xlim = [-0.2, 0.85]

high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #
inversion_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']


fig, ax = plt.subplots(figsize=(7,13))

for year, color in zip(['All', '2012', '2013', '2014', '2017'], ['black', 'blue', 'orange', 'green', 'purple']):

    if year == 'All':
        fitting_df_TRENDYv11_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}.csv')
        fitting_df_inversions_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}.csv')
        fitting_df_inversions_unscaled = fitting_df_inversions_unscaled.set_index('model_name').reindex(inversion_names).reset_index()
        fitting_df_NEEobservations_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_NEEobservations{lc_filestr}.csv')
        fitting_df_NEEobservations_unscaled.loc[fitting_df_NEEobservations_unscaled['model_name'] == 'FluxCOM-X-NEE','model_name'] = 'X-BASE'
        fitting_df_NEEobservations_unscaled.loc[fitting_df_NEEobservations_unscaled['model_name'] == 'ABCflux-NEE','model_name'] = 'ABCflux'
        fitting_df_reference_scaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_scaled_reference{lc_filestr}.csv')
        fitting_df_reference_scaled = fitting_df_reference_scaled.loc[fitting_df_reference_scaled['model_name'].isin(['APAR', 'GOME2_SIF']), :]  #'APAR', 'FPAR', 'LAI', 'PAR'
        fitting_df_reference_scaled.loc[fitting_df_reference_scaled['model_name'] == 'GOME2_SIF','model_name'] = 'SIF'

        # sort for each category
        fitting_df_TRENDYv11_sorted = fitting_df_TRENDYv11_unscaled.sort_values(f'{stat_var}')
        fitting_df_inversions_sorted = fitting_df_inversions_unscaled.sort_values(f'{stat_var}', na_position='first')
        fitting_df_NEEobservations_sorted = fitting_df_NEEobservations_unscaled.sort_values(f'{stat_var}')
        fitting_df_reference_sorted = fitting_df_reference_scaled.sort_values(f'{stat_var}')

        facecolors = 'None'# 'black'
        linewidths = 2
        
        results = OLSResults.load(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/regression/TRENDYv11_CLM5.0{lc_filestr}.pickle")
        n = results.summary2().tables[0].loc[3,1] # number of observations

        TRENDYv11_list = fitting_df_TRENDYv11_sorted['model_name'].tolist()
        inversions_list = fitting_df_inversions_sorted['model_name'].tolist()
        NEEobservations_list = fitting_df_NEEobservations_sorted['model_name'].tolist()
        reference_list = fitting_df_reference_sorted['model_name'].tolist()

    else:
        fitting_df_TRENDYv11_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_{year}.csv')
        fitting_df_inversions_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_{year}.csv')
        fitting_df_NEEobservations_unscaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_NEEobservations{lc_filestr}_{year}.csv')
        fitting_df_NEEobservations_unscaled.loc[fitting_df_NEEobservations_unscaled['model_name'] == 'FluxCOM-X-NEE','model_name'] = 'X-BASE'
        fitting_df_NEEobservations_unscaled.loc[fitting_df_NEEobservations_unscaled['model_name'] == 'ABCflux-NEE','model_name'] = 'ABCflux'
        fitting_df_reference_scaled = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_scaled_reference{lc_filestr}_{year}.csv')
        fitting_df_reference_scaled = fitting_df_reference_scaled.loc[fitting_df_reference_scaled['model_name'].isin(['APAR', 'GOME2_SIF']), :]  #'APAR', 'FPAR', 'LAI', 'PAR'
        fitting_df_reference_scaled.loc[fitting_df_reference_scaled['model_name'] == 'GOME2_SIF','model_name'] = 'SIF'

        # sort for each category
        fitting_df_TRENDYv11_sorted = fitting_df_TRENDYv11_unscaled.set_index('model_name').loc[TRENDYv11_list].reset_index()
        fitting_df_inversions_unscaled = fitting_df_inversions_unscaled.set_index('model_name').reindex(inversions_list).reset_index()
        fitting_df_inversions_sorted = fitting_df_inversions_unscaled.set_index('model_name').loc[inversions_list].reset_index()
        fitting_df_NEEobservations_sorted = fitting_df_NEEobservations_unscaled.set_index('model_name').loc[NEEobservations_list].reset_index()
        fitting_df_reference_sorted = fitting_df_reference_scaled.set_index('model_name').loc[reference_list].reset_index()

        facecolors = 'None'
        linewidths = 1
        
        start_month, end_month, campaign_name = get_campaign_info(int(year))
        df_airborne = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
        df_influence = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_regional_influence.csv')
        df = pd.concat((df_airborne, df_influence), axis=1)
        mask_id = np.where((df['background_CO2_std'].notna()) &
            # (local_hour.isin([13, 14, 15, 16])) &
            (df['ABoVE_influence_fraction'] > 0.5) &
            (df['ocean_influence_fraction'] < 0.3) &
            # (df['ABoVE_land_influence_fraction'] > 0.5)) and
            (df['CO2_change'] < 30) &
            (df['CO_change'] < 40))[0]
        n = len(mask_id)

    labelname = f"{year} (n={n})"

    fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(high_skill_TRENDY),'color'] = '#396bb8'
    fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin(low_skill_TRENDY),'color'] = '#d4631d'
    fitting_df_TRENDYv11_sorted.loc[fitting_df_TRENDYv11_sorted['model_name'].isin (['IBIS']),'color'] = 'grey' ## models with negative Rh

    plt.scatter(fitting_df_TRENDYv11_sorted[f'{stat_var}'], fitting_df_TRENDYv11_sorted['model_name'], marker='o', color=color, facecolors=facecolors, label=labelname, s=70, linewidths=linewidths)
    plt.scatter(fitting_df_NEEobservations_sorted[f'{stat_var}'], fitting_df_NEEobservations_sorted['model_name'], marker='o', color=color, facecolors=facecolors, s=70, linewidths=linewidths)
    plt.scatter(fitting_df_inversions_sorted[f'{stat_var}'], fitting_df_inversions_sorted['model_name'], marker='o', color=color, facecolors=facecolors, s=70, linewidths=linewidths)
    plt.scatter(fitting_df_reference_sorted[f'{stat_var}'], fitting_df_reference_sorted['model_name'], marker='o', color=color, facecolor=facecolors, s=70, linewidths=linewidths)
    
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
plt.axhline(y = fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')

plt.xlim(xlim)
plt.ylim(-1, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_reference_sorted.shape[0]-0.5)
plt.xlabel(r'Correlation with CO$_{2}$ observations', fontsize=18)
plt.xticks(ticks=np.arange(-0.2, 0.9, 0.1), fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)


colors = fitting_df_TRENDYv11_sorted['color'].values.tolist() + ['#6db753']*fitting_df_NEEobservations_unscaled.shape[0] + ['black']*fitting_df_inversions_sorted.shape[0] + ['purple']*fitting_df_reference_sorted.shape[0]
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)

ax.annotate("Remote Sensing", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_sorted.shape[0]+fitting_df_inversions_sorted.shape[0]+fitting_df_reference_sorted.shape[0]-1.5), fontsize=15)
ax.annotate("Atmospheric Inversions", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_unscaled.shape[0]+fitting_df_inversions_sorted.shape[0]-1.5), fontsize=15)
ax.annotate("Upscaled EC", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]+fitting_df_NEEobservations_unscaled.shape[0]-1.3), fontsize=15)
ax.annotate("TRENDY TBMs", (-0.18, fitting_df_TRENDYv11_sorted.shape[0]-1.5), fontsize=15)

plt.legend(bbox_to_anchor=(0.4, 0.75), fontsize=12)

fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig2_different_year.png', dpi=300, bbox_inches='tight')
fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig2_different_year.pdf', dpi=300, bbox_inches='tight')
plt.show()