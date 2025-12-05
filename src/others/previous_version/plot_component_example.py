
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read original simulated carbon fluxes
lcname = 'alllc' #alllc forest shrub tundra
lc_filestr = ''
weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'
dir0 = '/resnick/groups/carnegie_poc/jwen2/ABoVE/result/modify_NEE/'

seasonal_df_TRENDYv11NEE = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11GPP = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11GPP_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Ra = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Ra_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Rh = pd.read_csv(f"/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Rh_{regionname}_{lcname}_{weightname}.csv")

# only select growing seasons (Apr-Nov)
seasonal_df_TRENDYv11NEE = seasonal_df_TRENDYv11NEE.loc[3:10]
seasonal_df_TRENDYv11GPP = seasonal_df_TRENDYv11GPP.loc[3:10]
seasonal_df_TRENDYv11Ra = seasonal_df_TRENDYv11Ra.loc[3:10]
seasonal_df_TRENDYv11Rh = seasonal_df_TRENDYv11Rh.loc[3:10]


# plot
model_name = 'ISBA-CTRIP' # ISBA-CTRIP CABLE-POP
seasonal_GPP_model = seasonal_df_TRENDYv11GPP[model_name]
seasonal_Ra_model = seasonal_df_TRENDYv11Ra[model_name]
seasonal_Rh_model = seasonal_df_TRENDYv11Rh[model_name]
seasonal_NEE_model = seasonal_df_TRENDYv11NEE[model_name]

# plot data

for (varname, seasonal_df, color) in zip(['GPP', 'Ra', 'Rh', 'NEE'],
                                         (seasonal_GPP_model, seasonal_Ra_model, seasonal_Rh_model, seasonal_NEE_model), 
                                         ['#438382', '#c0ac1a', '#a68179', '#d4631d']):
    
    if model_name == 'ISBA-CTRIP' and varname == 'NEE':
        color = '#396bb8'

    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(np.arange(4, 12), seasonal_df, linestyle='-', color=color, linewidth=3, label='GPP')
    ax.set_xlim(4, 11) #4,11
    ax.set_ylim(-1.5, 4)
    ax.set_xticks(np.arange(4, 12))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Carbon fluxes\n(μmol m$^{-2}$ s$^{-1}$)', fontsize=16, labelpad=0)