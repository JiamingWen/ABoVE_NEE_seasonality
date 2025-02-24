# compare performance of inversions with in-situ data and satellite data as input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stat_var = 'cor'; xlim = [0.55, 0.8]

# read model performance
fitting_df_all = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE.csv')
fitting_df_2017 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE_2017.csv')
fitting_df_inversions_unscaled = pd.merge(fitting_df_all[['model_name', 'cor']], fitting_df_2017[['model_name', 'cor']], on='model_name', how='outer', suffixes=('_All', '_2017'))
fitting_df_inversions_unscaled.loc[fitting_df_inversions_unscaled['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_inversions_unscaled.rename(columns={'cor_All': 'All', 'cor_2017': '2017'}, inplace=True)
inversion_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
fitting_df_inversions_unscaled.fillna(-999, inplace=True)

# data input
inv_insitu_list = ['CAMS', 'CarboScope', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC4-ACTM', 'NISMON-CO2', 'UoE']
inv_satellite_list = ['CAMS-Satellite', 'GCASv2', 'GONGGA', 'THU']
inv_combine_list = ['CMS-Flux', 'COLA']

fitting_df_insitu = fitting_df_inversions_unscaled[fitting_df_inversions_unscaled['model_name'].isin(inv_insitu_list)]
fitting_df_satellite = fitting_df_inversions_unscaled[fitting_df_inversions_unscaled['model_name'].isin(inv_satellite_list)]
fitting_df_combine = fitting_df_inversions_unscaled[fitting_df_inversions_unscaled['model_name'].isin(inv_combine_list)]

# in alphabetical order
fitting_df_insitu_sorted = fitting_df_insitu.sort_values('model_name', ascending=False)
fitting_df_satellite_sorted = fitting_df_satellite.sort_values('model_name', ascending=False)
fitting_df_combine_sorted = fitting_df_combine.sort_values('model_name', ascending=False)


# plot
fig, ax = plt.subplots(1, 2, figsize=(11.5,5))
subplot_id = 0

for year, color in zip(['All', '2017'], ['black', 'purple']):

    subplot_id += 1
    ax1 = plt.subplot(1, 2, subplot_id)

    if year == 'All':
        facecolors = 'None'# 'black'
        linewidths = 1
        column_name = 'All'
        subtitle = '(a) All years'

    else:
        facecolors = 'None'
        linewidths = 1
        column_name = year
        subtitle = '(b) 2017'

    plt.scatter(fitting_df_combine_sorted[column_name], fitting_df_combine_sorted['model_name'], marker='o', color=color, facecolors=facecolors, s=70, linewidths=linewidths)
    plt.scatter(fitting_df_satellite_sorted[column_name], fitting_df_satellite_sorted['model_name'], marker='o', color=color, facecolors=facecolors, s=70, linewidths=linewidths)
    plt.scatter(fitting_df_insitu_sorted[column_name], fitting_df_insitu_sorted['model_name'], marker='o', color=color, facecolors=facecolors, s=70, linewidths=linewidths)

    plt.axhline(y = fitting_df_combine_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')
    plt.axhline(y = fitting_df_combine_sorted.shape[0]+fitting_df_satellite_sorted.shape[0]-0.5, color = 'grey', linestyle = '--')

    plt.xlim(xlim)
    plt.ylim(-0.5, fitting_df_insitu_sorted.shape[0]+fitting_df_satellite_sorted.shape[0]+fitting_df_combine_sorted.shape[0]-0.5)
    plt.xlabel(r'Correlation with CO$_{2}$ observations', fontsize=18)
    plt.xticks(ticks=np.arange(0.55, 0.85, 0.05), fontsize=15) #np.arange(-0.2, 1, 0.2), 
    plt.yticks(fontsize=15)


    ax1.annotate("In situ", (0.56, fitting_df_combine_sorted.shape[0]+fitting_df_satellite_sorted.shape[0]+fitting_df_insitu_sorted.shape[0]-1.3), fontsize=15)
    ax1.annotate("Satellite", (0.56, fitting_df_combine_sorted.shape[0]+fitting_df_satellite_sorted.shape[0]-1.3), fontsize=15)
    ax1.annotate("Combined", (0.56, fitting_df_combine_sorted.shape[0]-1.3), fontsize=15)

    ax1.annotate(subtitle, xy=(0.95, 0.97), xycoords='axes fraction', fontsize=18, ha='right', va='top')

plt.subplots_adjust(wspace=0.1)
plt.tight_layout()

plt.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/compare_inv_insitu_satellite.png', dpi=300, bbox_inches='tight')
plt.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/compare_inv_insitu_satellite.pdf', dpi=300, bbox_inches='tight')
plt.show()
