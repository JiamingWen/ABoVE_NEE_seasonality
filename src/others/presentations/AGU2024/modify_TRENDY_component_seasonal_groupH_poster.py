import numpy as np
import pandas as pd
import os
os.chdir('/resnick/groups/carnegie_poc/jwen2/ABoVE/src')
from functions import get_campaign_info
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


lcname = 'alllc' #alllc forest shrub tundra
lc_filestr = ''
weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'
dir0 = '/resnick/groups/carnegie_poc/jwen2/ABoVE/result/modify_NEE/'

# model performance with original seasonal cycle
fitting_df_TRENDYv11_unscaled_only_seasonal = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_TRENDYv11_only_seasonal.csv')
fitting_df_TRENDYv11_unscaled_only_seasonal = fitting_df_TRENDYv11_unscaled_only_seasonal.loc[~fitting_df_TRENDYv11_unscaled_only_seasonal['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted = fitting_df_TRENDYv11_unscaled_only_seasonal.sort_values('cor')
high_model_subset = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['cor']>0.63, 'model_name'].tolist()
# high_model_subset.remove("IBIS")
low_model_subset = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['cor']<0.63, 'model_name'].tolist()
# high_model_subset = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0'] # exclude IBIS
# low_model_subset = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM']

# colors for making plots
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(high_model_subset),'color'] = '#5986cb'
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted.loc[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(low_model_subset),'color'] = '#e57f3f'


'''plot in four separate panels'''
fig, ax = plt.subplots(4,1,figsize=(7,28))

colors = ['#40B0A6', '#D35FB7', '#ff6666', '#1A85FF']
markers = ['d', 'd', 'd', 'd'] # markers = ['d', 'v', '^', 's']
sizes = [100, 100, 100, 100]  # sizes = [80, 80, 80, 50]
labels = ['GPP seasonality', 'Ra seasonality', 'Rh seasonality', 'Relative magnitudes']
subtitles = ['(a)', '(b)', '(c)', '(d)']
case_id = [2, 3, 4, 1]

for plot_id in np.arange(4):
    ax_sub = plt.subplot(4,1,plot_id+1)
    cor_modified_case = pd.read_csv(f"{dir0}cor_modified_case{case_id[plot_id]}_groupH.csv")
    cor_modified_case = cor_modified_case[low_model_subset]
    cor_modified_case_median = cor_modified_case.median(axis=0)
    cor_modified_case_min = cor_modified_case.min(axis=0)
    cor_modified_case_max = cor_modified_case.max(axis=0)
    plt.scatter(cor_modified_case_median, np.arange(len(low_model_subset)), marker=markers[plot_id], color=colors[plot_id], s=sizes[plot_id], label=labels[plot_id])
    plt.errorbar((cor_modified_case_max+cor_modified_case_min)/2, np.arange(len(low_model_subset)), xerr=(cor_modified_case_max-cor_modified_case_min)/2, ecolor=colors[plot_id], fmt='none', alpha=0.3)
    
    # plot original correlation
    fitting_df_TRENDYv11_low = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted[fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'].isin(low_model_subset)]
    plt.scatter(fitting_df_TRENDYv11_low['cor'], np.arange(len(low_model_subset)), marker='x', color='#e57f3f', s=100)
    
    # add reference ribbons
    # fitting_df_regression_scaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_regression{lc_filestr}.csv')
    # plt.axvspan(fitting_df_regression_scaled.loc[fitting_df_regression_scaled['model_name']=='constant','cor_CI_low'].values[0], fitting_df_regression_scaled.loc[fitting_df_regression_scaled['model_name']=='constant','cor_CI_high'].values[0], alpha=0.2, color='olive')
    fitting_df_reference_scaled_only_seasonal = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_reference_only_seasonal{lc_filestr}.csv')
    plt.axvspan(fitting_df_reference_scaled_only_seasonal.loc[fitting_df_reference_scaled_only_seasonal['model_name']=='APAR','cor_CI_low'].values[0], fitting_df_reference_scaled_only_seasonal.loc[fitting_df_reference_scaled_only_seasonal['model_name']=='APAR','cor_CI_high'].values[0], alpha=0.2, color='purple')

    # add boxplots
    plt.boxplot(fitting_df_TRENDYv11_low['cor'], positions=[-1], widths=0.5, vert=False, patch_artist=True, 
                boxprops=dict(facecolor='#e57f3f', color='#e57f3f'), 
                medianprops=dict(color='black', linewidth=2), flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': '#e57f3f', 'markeredgecolor': 'none'})
    plt.boxplot(cor_modified_case_median, positions=[-2], widths=0.5, vert=False, patch_artist=True, 
                boxprops=dict(facecolor=colors[plot_id], color=colors[plot_id]), 
                medianprops=dict(color='black', linewidth=2), flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': colors[plot_id], 'markeredgecolor': 'none'})

    # plot settings
    plt.xlim([-0.1, 0.75])
    plt.xlabel(r'Correlation with CO$_{2}$ observations', fontsize=18)
    plt.xticks(ticks=np.arange(-0.1, 0.8, 0.1), fontsize=15)

    plt.ylim([-2.5, len(low_model_subset)-0.5])
    plt.yticks(fontsize=18)
    yticklabels = ['Adjusted', 'Original'] + low_model_subset
    ytick_colors = ['black', 'black'] + ['#e57f3f']*len(low_model_subset)
    ax_sub.set_yticks(np.arange(len(yticklabels))-2)
    ax_sub.set_yticklabels(yticklabels)
    for ytick, color in zip(ax_sub.get_yticklabels(), ytick_colors):
        ytick.set_color(color)
    # ax_sub.tick_params(axis='y', colors='black')

    plt.axhline(y=-0.5, color='black', linestyle='--', linewidth=1)
    plt.text(0.05, 0.95, f"{subtitles[plot_id]} {labels[plot_id]}", transform=ax_sub.transAxes, fontsize=20, verticalalignment='top')

plt.subplots_adjust(wspace=0.3, hspace=0.2)
plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/figures/modify_TRENDY_component_seasonal_groupH_poster.png', dpi=300, bbox_inches='tight')
plt.show()
