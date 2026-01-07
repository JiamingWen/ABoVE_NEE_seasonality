'''plot statistics between observed and modeled CO2 enhancements'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stat_var = 'cor'; xlim = [-0.2, 0.8]; xlabel = r'Correlation'
# stat_var = 'mean_bias'; xlim = [-8, 2]; xlabel = r'Mean bias (ppm)'
# stat_var = 'range_ratio_95_5'; xlim = [0, 3]; xlabel = r'Range ratio'

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

# # statistics
# fitting_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_background-ct.csv')
# # fitting_df_TRENDYv11 = fitting_df_TRENDYv11.loc[~fitting_df_TRENDYv11['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
# fitting_df_inversions = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_background-ct.csv')
# fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
# fitting_df_UpscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_background-ct.csv')


# fitting_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_background-ct_diurnal_x_base.csv')
# # fitting_df_TRENDYv11 = fitting_df_TRENDYv11.loc[~fitting_df_TRENDYv11['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
# fitting_df_inversions = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_background-ct_diurnal_x_base.csv')
# fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
# fitting_df_UpscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_background-ct_diurnal_x_base.csv')


if stat_var in ['mean_bias', 'range_ratio_95_5']: # imposing X-BASE diurnal cycle
    fitting_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_background-ct.csv')
    # fitting_df_TRENDYv11 = fitting_df_TRENDYv11.loc[~fitting_df_TRENDYv11['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
    fitting_df_inversions = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_background-ct.csv')
    fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
    fitting_df_UpscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_background-ct.csv')
else:
    fitting_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}.csv')
    # fitting_df_TRENDYv11 = fitting_df_TRENDYv11.loc[~fitting_df_TRENDYv11['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
    fitting_df_inversions = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}.csv')
    fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
    fitting_df_UpscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}.csv')

# Combine dataframes into a single dataframe with a 'group' column
fitting_df_TRENDYv11['group'] = 'TRENDY TBMs'
fitting_df_inversions['group'] = 'Inversions'
fitting_df_UpscaledEC['group'] = 'Upscaled fluxes'
combined_data = pd.concat([fitting_df_TRENDYv11, fitting_df_inversions, fitting_df_UpscaledEC])

# Plot boxplots and overlay individual points
fig, ax = plt.subplots(figsize=(5, 3))
# Define colors for groups and subgroups
colors = {'Inversions': 'black', 'Upscaled fluxes': '#4c8638', 'TRENDY TBMs': '#d4631d'} # #56983f

# Plot horizontal boxplots at specified y positions
boxprops = dict(vert=False, patch_artist=False, widths=0.6, showcaps=True, showfliers=False)

bp_inv = ax.boxplot([fitting_df_inversions[stat_var].dropna().values], positions=[3], **boxprops)
bp_tr = ax.boxplot([fitting_df_TRENDYv11[stat_var].dropna().values], positions=[1], **boxprops)

# Customize boxplot colors
for box, color in zip([bp_tr['boxes'][0], bp_inv['boxes'][0]], ['#d4631d', 'black']):
    box.set(color=color, linewidth=2)
    if box is bp_tr['boxes'][0]:
        bp = bp_tr
    else:
        bp = bp_inv
    bp['medians'][0].set(color=color, linewidth=2)
    for whisker in bp['whiskers']:
        whisker.set(color=color, linewidth=1.5)
    for cap in bp['caps']:
        cap.set(color=color, linewidth=1.5)

# Overlay individual data points
for group, group_data in combined_data.groupby('group'):
    y_pos = {'TRENDY TBMs': 1, 'Upscaled fluxes': 2, 'Inversions': 3}[group]
    for _, row in group_data.iterrows():
        ax.plot(row[stat_var], y_pos, 'o', markeredgecolor=colors[group], markerfacecolor='none', alpha=0.9)

# Customize plot
ax.tick_params(axis='x', labelsize=14)
ax.set_xlim(xlim)
if stat_var == 'range_ratio_95_5':
    ax.set_xticks(np.arange(0, 3.5, 0.5))
ax.set_ylim(0.5, 3.5)
ax.set_yticks([1, 2, 3])
ax.yaxis.set_ticks_position('left')
ax.tick_params(axis='y', pad=25)  # move tick labels further left
ax.spines['left'].set_position(('outward', 25))  # move (hidden) left spine outward to help label placement
ax.tick_params(axis='y', length=0)  # Remove tick symbols
ax.set_yticklabels(['TRENDY\n  TBMs', 'Upscaled\n  fluxes', 'Atmospheric\n  Inversions'], fontsize=14, ha='center')
ax.set_xlabel(xlabel, fontsize=14)
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Remove box boundary
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

plt.suptitle('')  # Remove automatic title added by pandas boxplot
plt.tight_layout()
plt.show()

