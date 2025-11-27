''' compare relative magnitude of component fluxes '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lcname = 'alllc' #alllc forest shrub tundra
lc_filestr = ''
weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'

# model performance with original seasonal cycle
fitting_df_TRENDYv11_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11_only_seasonal.csv')
fitting_df_TRENDYv11_unscaled_only_seasonal = fitting_df_TRENDYv11_unscaled_only_seasonal.loc[~fitting_df_TRENDYv11_unscaled_only_seasonal['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_TRENDYv11_unscaled_only_seasonal_sorted = fitting_df_TRENDYv11_unscaled_only_seasonal.sort_values('cor')

# read original simulated carbon fluxes
seasonal_df_TRENDYv11NEE = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11GPP = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11GPP_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Ra = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Ra_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Rh = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Rh_{regionname}_{lcname}_{weightname}.csv")

# only select growing seasons (Apr-Nov)
seasonal_df_TRENDYv11NEE = seasonal_df_TRENDYv11NEE.loc[3:10]
seasonal_df_TRENDYv11GPP = seasonal_df_TRENDYv11GPP.loc[3:10]
seasonal_df_TRENDYv11Ra = seasonal_df_TRENDYv11Ra.loc[3:10]
seasonal_df_TRENDYv11Rh = seasonal_df_TRENDYv11Rh.loc[3:10]

annual_sum_GPP = seasonal_df_TRENDYv11GPP.sum(axis=0)
annual_sum_Ra = seasonal_df_TRENDYv11Ra.sum(axis=0)
annual_sum_Rh = seasonal_df_TRENDYv11Rh.sum(axis=0)
annual_sum_NEE = seasonal_df_TRENDYv11NEE.sum(axis=0)

# plot a stacked barplot

# unsorted
# x = seasonal_df_TRENDYv11NEE.columns
# y1 = annual_sum_Ra / annual_sum_GPP
# y2 = annual_sum_Rh / annual_sum_GPP
# y3 = -annual_sum_NEE / annual_sum_GPP

# in a order of model performance
# model_sorted = fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'][::-1]

# in alphabetical order
model_sorted = sorted(fitting_df_TRENDYv11_unscaled_only_seasonal_sorted['model_name'])

x = model_sorted
y1 = annual_sum_Ra[model_sorted] / annual_sum_GPP[model_sorted]
y2 = annual_sum_Rh[model_sorted] / annual_sum_GPP[model_sorted]
y3 = -annual_sum_NEE[model_sorted] / annual_sum_GPP[model_sorted]

# make plots 
fig, ax = plt.subplots(figsize=(5, 4))
plt.bar(x, y1, color='#c0ac1a', label='$\it{R}_{a}$', alpha=0.9)
plt.bar(x, y2, bottom=y1, color='#e573d5', label='$\it{R}_{h}$', alpha=0.9)
# plt.bar(x, y3, bottom=y1+y2, color='y', label='-NEE')
# plt.xlabel("Models")
plt.ylabel("Ratio to GPP", fontsize=14)
plt.xticks(rotation=90)
ax.set_yticks(np.arange(0, 1.2, 0.2), ['0%', '20%', '40%', '60%', '80%', '100%'])
plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
plt.ylim([0, 1.15])
plt.legend(ncol=2, bbox_to_anchor=(0, 1), loc='upper left')

fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/compare_component_flux_ratio.png', dpi=300, bbox_inches='tight')
fig.savefig('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/compare_component_flux_ratio.pdf', dpi=300, bbox_inches='tight')
plt.show()