'''plot an example of four types of adjustment'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read original simulated carbon fluxes
lcname = 'alllc' #alllc forest shrub tundra
lc_filestr = ''
weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'
dir0 = '/central/groups/carnegie_poc/jwen2/ABoVE/result/modify_NEE/'

seasonal_df_TRENDYv11NEE = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11GPP = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11GPP_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Ra = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Ra_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Rh = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Rh_{regionname}_{lcname}_{weightname}.csv")

# only select growing seasons (Apr-Nov)
seasonal_df_TRENDYv11NEE = seasonal_df_TRENDYv11NEE.loc[3:10]
seasonal_df_TRENDYv11GPP = seasonal_df_TRENDYv11GPP.loc[3:10]
seasonal_df_TRENDYv11Ra = seasonal_df_TRENDYv11Ra.loc[3:10]
seasonal_df_TRENDYv11Rh = seasonal_df_TRENDYv11Rh.loc[3:10]


''' calculate annual or growing season sum'''
def calculate_annual_sum (mean_seasonal_cycle):
    return np.sum(mean_seasonal_cycle)

''' modify magnitude of carbon flux while keeping seasonality unchanged'''
def modify_magnitude (mean_seasonal_cycle_model, mean_seasonal_cycle_ref):
    result = mean_seasonal_cycle_model / calculate_annual_sum(mean_seasonal_cycle_model) * calculate_annual_sum(mean_seasonal_cycle_ref)
    return result

''' modify seasonality of carbon flux while keeping annual sum unchanged'''
def modify_seasonality (mean_seasonal_cycle_model, mean_seasonal_cycle_ref):
    result = mean_seasonal_cycle_ref / calculate_annual_sum(mean_seasonal_cycle_ref) * calculate_annual_sum(mean_seasonal_cycle_model)
    return result

# standardize with minumum NEE
def scale_minumum (vec):
    return -vec / np.min(vec)

# plot
model_name = 'CABLE-POP'
seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
seasonal_Ra_model = seasonal_df_TRENDYv11Ra[model_name]
seasonal_Rh_model = seasonal_df_TRENDYv11Rh[model_name]
seasonal_NEE_model = seasonal_df_TRENDYv11NEE[model_name]

model_name_ref = 'ISBA-CTRIP'
seasonal_GPP_ref = seasonal_df_TRENDYv11GPP[model_name_ref]
seasonal_Ra_ref = seasonal_df_TRENDYv11Ra[model_name_ref]
seasonal_Rh_ref = seasonal_df_TRENDYv11Rh[model_name_ref]


labels_adjustment = ['Adjusting GPP seasonality', 'Adjusting $\it{R}_{a}$ seasonality', 'Adjusting $\it{R}_{h}$ seasonality', 'Adjusting relative proportion']
colors = ['#50c878', '#d4bf1d', '#1367e9']
subtitles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

fig, ax = plt.subplots(4, 2, figsize=(10, 18))
for plot_id in np.arange(4):

    # fig, ax = plt.subplots(figsize=(4, 4))
    ax_sub = plt.subplot(4, 2, plot_id*2+1)

    # adjust NEE
    if plot_id == 0: # adjust GPP seasonality
        seasonal_GPP_model_modified = modify_seasonality(seasonal_GPP_model, seasonal_GPP_ref)
        seasonal_Ra_model_modified = seasonal_Ra_model
        seasonal_Rh_model_modified = seasonal_Rh_model
        alpha_modified = [1, 0, 0, 1]
        
    elif plot_id == 1: # adjust Ra seasonality
        seasonal_GPP_model_modified = seasonal_GPP_model
        seasonal_Ra_model_modified = modify_seasonality(seasonal_Ra_model, seasonal_Ra_ref)
        seasonal_Rh_model_modified = seasonal_Rh_model
        alpha_modified = [0, 1, 0, 1]

    elif plot_id == 2: # adjust Rh seasonality
        seasonal_GPP_model_modified = seasonal_GPP_model
        seasonal_Ra_model_modified = seasonal_Ra_model
        seasonal_Rh_model_modified = modify_seasonality(seasonal_Rh_model, seasonal_Rh_ref)
        alpha_modified = [0, 0, 1, 1]

    elif plot_id == 3: # adjust component share
        seasonal_GPP_model_modified = seasonal_GPP_model
        seasonal_Ra_model_modified = modify_magnitude(seasonal_Ra_model, seasonal_Ra_ref) / sum(seasonal_GPP_ref) * sum(seasonal_GPP_model)
        seasonal_Rh_model_modified = modify_magnitude(seasonal_Rh_model, seasonal_Rh_ref) / sum(seasonal_GPP_ref) * sum(seasonal_GPP_model)
        alpha_modified = [0, 1, 1, 1]

    seasonal_NEE_model_modified = seasonal_Ra_model_modified + seasonal_Rh_model_modified - seasonal_GPP_model_modified


    # plot original data
    ax_sub.plot(np.arange(4, 12), seasonal_GPP_model, linestyle='-', color=colors[0], linewidth=3, alpha=0.5, label='GPP')
    ax_sub.plot(np.arange(4, 12), seasonal_Ra_model, linestyle='-', color=colors[1], linewidth=3, alpha=0.5, label='$\it{R}_{a}$')
    ax_sub.plot(np.arange(4, 12), seasonal_Rh_model, linestyle='-', color=colors[2], linewidth=3, alpha=0.5, label='$\it{R}_{h}$')

    # plot adjusted data
    ax_sub.plot(np.arange(4, 12), seasonal_GPP_model_modified, linestyle='--', color=colors[0], linewidth=3, alpha=alpha_modified[0])
    ax_sub.plot(np.arange(4, 12), seasonal_Ra_model_modified, linestyle='--', color=colors[1], linewidth=3, alpha=alpha_modified[1])
    ax_sub.plot(np.arange(4, 12), seasonal_Rh_model_modified, linestyle='--', color=colors[2], linewidth=3, alpha=alpha_modified[2])

    # plot settings
    ax_sub.set_xlim(4, 11) #4,11
    ax_sub.set_ylim(0, 5)
    ax_sub.set_xticks(np.arange(4, 12))
    ax_sub.tick_params(axis='x', labelsize=14)
    ax_sub.tick_params(axis='y', labelsize=14)
    ax_sub.set_xlabel('Month', fontsize=16)
    ax_sub.set_ylabel('Carbon fluxes\n(μmol m$^{-2}$ s$^{-1}$)', fontsize=16, labelpad=0)
    plt.text(0.05, 0.95, f"{subtitles[plot_id*2]} {labels_adjustment[plot_id]}", transform=ax_sub.transAxes, fontsize=14, verticalalignment='top')
    
    if plot_id == 0:
        handles, labels = ax_sub.get_legend_handles_labels()
        ax_sub.legend(handles, labels, loc='upper center', ncol=1, fontsize=14, bbox_to_anchor=(0.8, 0.85))

    # plot NEE separately
    ax_sub = plt.subplot(4, 2, plot_id*2+2)
    ax_sub.plot(np.arange(4, 12), scale_minumum(seasonal_NEE_model), linestyle='-', color='#e57f3f', linewidth=3, alpha=0.5, label='NEE')
    ax_sub.plot(np.arange(4, 12), scale_minumum(seasonal_NEE_model_modified), linestyle='--', color='#e57f3f', linewidth=3, alpha=1) #, label='NEE modified'
    
    # plot settings
    ax_sub.set_xlim(4, 11) #4,11
    ax_sub.set_ylim(-1.2, 1.2)
    ax_sub.set_xticks(np.arange(4, 12))
    ax_sub.tick_params(axis='x', labelsize=14)
    ax_sub.tick_params(axis='y', labelsize=14)
    ax_sub.set_xlabel('Month', fontsize=16)
    ax_sub.set_ylabel('Rescaled NEE', fontsize=16, labelpad=0)
    plt.text(0.05, 0.95, f"{subtitles[plot_id*2+1]} {labels_adjustment[plot_id]}", transform=ax_sub.transAxes, fontsize=14, verticalalignment='top')

    if plot_id == 0:
        handles, labels = ax_sub.get_legend_handles_labels()
        ax_sub.legend(handles, labels, loc='upper center', ncol=1, fontsize=14, bbox_to_anchor=(0.2, 0.85))

# Add legend for line styles
solid_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='-', linewidth=3, alpha=0.5)
dashed_line = plt.Line2D((0, 1), (0, 0), color='black', linestyle='--', linewidth=3)
fig.legend([solid_line, dashed_line], ['Original data', 'Adjusted data'], loc='upper center', ncol=2, fontsize=14, bbox_to_anchor=(0.5, 0.92))

plt.subplots_adjust(wspace=0.3, hspace=0.2)
plt.show()

fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/modify_TRENDY_component_seasonal_example.png', dpi=300, bbox_inches='tight')
fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/modify_TRENDY_component_seasonal_example.pdf', dpi=300, bbox_inches='tight')
