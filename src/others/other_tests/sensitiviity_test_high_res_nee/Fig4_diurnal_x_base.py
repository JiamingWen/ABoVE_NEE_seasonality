'''similar to Fig. 4, but imposing the X-BASE diurnal cycle for all points'''

'''
plot change in correlation of lower-correlation TBMs
after modify the magnitude and seasonal cycle of component fluxes
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lcname = 'alllc' #alllc forest shrub tundra
lc_filestr = ''
weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'
dir0 = '/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/seasonality_adjustment/'
low_model_subset = sorted(['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'], reverse=True)
fitting_df_TRENDYv11_unscaled_only_seasonal = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11_only_seasonal_diurnal_x_base.csv')
fitting_df_TRENDYv11_low = fitting_df_TRENDYv11_unscaled_only_seasonal[fitting_df_TRENDYv11_unscaled_only_seasonal['model_name'].isin(low_model_subset)]
fitting_df_TRENDYv11_low = fitting_df_TRENDYv11_low.sort_values('model_name', ascending=False)

'''plot in four separate panels'''
fig, ax = plt.subplots(2, 2, figsize=(16, 15))

colors = ['#438382', '#c0ac1a', '#e573d5', '#a68179', 'grey']

markers = ['x', 'x', 'x', 'x', 'x']
sizes = [80, 80, 80, 80, 80]
labels = ['GPP seasonality', '$\it{R}_{a}$ seasonality', '$\it{R}_{h}$ seasonality', 'Relative proportion', 'Reco seasonality']
subtitles = ['(a)', '(b)', '(c)', '(d)', '(e)']
case_str = ['GPP_seasonality', 'Ra_seasonality', 'Rh_seasonality', 'relative_proportion', 'Reco_seasonality']

for plot_id in np.arange(4):
    ax_sub = plt.subplot(2, 2,plot_id+1)

    # add reference ribbons
    fitting_df_reference_scaled_only_seasonal = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_scaled_reference_only_seasonal{lc_filestr}.csv')
    # plt.axvspan(fitting_df_reference_scaled_only_seasonal.loc[fitting_df_reference_scaled_only_seasonal['model_name']=='APAR','cor_CI_low'].values[0], fitting_df_reference_scaled_only_seasonal.loc[fitting_df_reference_scaled_only_seasonal['model_name']=='APAR','cor_CI_high'].values[0], alpha=0.2, color='purple')
    plt.axvline(x=fitting_df_reference_scaled_only_seasonal.loc[fitting_df_reference_scaled_only_seasonal['model_name']=='APAR','cor'].values[0], color='purple', linestyle='--', linewidth=3, alpha=0.9)

    # plot original correlation
    plt.scatter(fitting_df_TRENDYv11_low['cor'], np.arange(len(low_model_subset)), marker='o', color='#D4631D', s=70, label='Original')
    
    # plot adjusted correlation
    cor_modified_case = pd.read_csv(f"{dir0}cor_modified_{case_str[plot_id]}_groupH_x_base_diurnal.csv")
    cor_modified_case = cor_modified_case[low_model_subset]
    cor_modified_case_median = cor_modified_case.median(axis=0)
    cor_modified_case_min = cor_modified_case.min(axis=0)
    cor_modified_case_max = cor_modified_case.max(axis=0)
    plt.scatter(cor_modified_case_median, np.arange(len(low_model_subset)), marker=markers[plot_id], color=colors[plot_id], s=sizes[plot_id], label='Adjusted', edgecolor='black', linewidth=3)
    plt.errorbar((cor_modified_case_max+cor_modified_case_min)/2, np.arange(len(low_model_subset)), xerr=(cor_modified_case_max-cor_modified_case_min)/2, ecolor=colors[plot_id], fmt='none', alpha=0.5)
    
    # add boxplots
    plt.boxplot(fitting_df_TRENDYv11_low['cor'], positions=[-1], widths=0.5, vert=False, patch_artist=True, 
                boxprops=dict(facecolor='#D4631D', color='#D4631D'), 
                medianprops=dict(color='black', linewidth=5), flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': '#D4631D', 'markeredgecolor': 'none'})
    plt.boxplot(cor_modified_case_median, positions=[-2], widths=0.5, vert=False, patch_artist=True, 
                boxprops=dict(facecolor=colors[plot_id], color=colors[plot_id]), 
                medianprops=dict(color='black', linewidth=5), flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': colors[plot_id], 'markeredgecolor': 'none'})

    # plot settings
    xlim = [-0.1, 0.75]
    plt.xlim(xlim)
    plt.xlabel(r'Correlation with CO$_{2}$ observations', fontsize=18)
    plt.xticks(ticks=np.arange(xlim[0], xlim[1], 0.1), fontsize=15)

    plt.ylim([-2.5, len(low_model_subset)-0.5])
    plt.yticks(fontsize=18)
    yticklabels = ['Adjusted', 'Original'] + fitting_df_TRENDYv11_low['model_name'].values.tolist()
    ytick_colors = ['black', 'black'] + ['#D4631D']*len(low_model_subset)
    ax_sub.set_yticks(np.arange(len(yticklabels))-2)
    ax_sub.set_yticklabels(yticklabels)
    for ytick, color in zip(ax_sub.get_yticklabels(), ytick_colors):
        ytick.set_color(color)
    # ax_sub.tick_params(axis='y', colors='black')

    plt.axhline(y=-0.5, color='black', linestyle='--', linewidth=1)
    plt.text(0.05, 0.95, f"{subtitles[plot_id]} {labels[plot_id]}", transform=ax_sub.transAxes, fontsize=20, verticalalignment='top')

    # legend
    plt.legend(fontsize=15, loc='center left')

plt.subplots_adjust(wspace=0.3, hspace=0.2)

plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/seasonality_adjustment/Fig4_diurnal_x_base.png', dpi=300, bbox_inches='tight')
plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_high_res_nee/seasonality_adjustment/Fig4_diurnal_x_base.pdf', dpi=300, bbox_inches='tight')
plt.show()