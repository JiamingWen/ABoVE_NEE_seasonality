'''compare seasonal variations of tsl, nee, npp, rh for the selected TRENDY models '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/src')
from functions import get_campaign_info

regionname = 'ABoVEcore'
weightname = 'unweighted' #unweighted weighted
lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

# standardize with maximum
def scale_maximum (vec):
    return vec / np.max(abs(vec))

# standardize with maximum and minimum
def scale_maximum_minimum (vec):
    return (vec - np.min(vec)) / (np.max(vec) - np.min(vec))

# no standardization
def no_scale (vec):
    return vec-273.25

models = ['ISBA-CTRIP', 'CLASSIC', 'JSBACH', 'JULES']
colors = ['#5986cb','#5986cb','#e57f3f','#e57f3f']
linestyles = ['dashed','dotted','dashed','dotted']
linewidths = [2,3,2,3]

fig, ax = plt.subplots(2, 2, figsize=(8, 7))
for j, varname in enumerate(['NEE', 'NPP', 'Rh', 'tsl']):
    ax1 = plt.subplot(2, 2, j+1)

    # some settings
    if varname == 'NEE':
        filestr = ''
        scale_fun = scale_maximum
        ylim = [-1.2,1.2]
        ylabel = f'Standardized {varname}'
    elif varname in ['NPP', 'Rh']:
        filestr = varname
        scale_fun = scale_maximum_minimum
        ylim = [0,1.2]
        ylabel = f'Standardized {varname}'
    elif varname == 'tsl':
        filestr = varname
        scale_fun = no_scale
        ylim = [-20, 25]
        ylabel = 'Soil temperature (\u00b0C)'

    # read data
    if varname in ['NEE', 'Rh', 'tsl']:
        seasonal_df_TRENDYv11 = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_TRENDYv11{filestr}_{regionname}_{lcname}_{weightname}.csv")
    else:
        seasonal_df_TRENDYv11GPP = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_TRENDYv11GPP_{regionname}_{lcname}_{weightname}.csv")
        seasonal_df_TRENDYv11Ra = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_TRENDYv11Ra_{regionname}_{lcname}_{weightname}.csv")
        seasonal_df_TRENDYv11 = seasonal_df_TRENDYv11GPP - seasonal_df_TRENDYv11Ra

    seasonal_df_TRENDYv11 = seasonal_df_TRENDYv11.apply(scale_fun, axis=0)

    # plot
    for i, model_name in enumerate(models):

        color = colors[i]
        linestyle = linestyles[i]
        linewidth = linewidths[i]

        plt.plot(np.arange(1,13),seasonal_df_TRENDYv11[model_name], linestyle=linestyle, color=color, linewidth=linewidth, label=model_name)
        
    # plt.title(titlename, fontsize=20)
    plt.xlim(4,11)
    plt.ylim(ylim)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax1.set_xticks(np.arange(4,12))
    plt.xlabel('Month', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    subtitle = chr(ord('`')+j+1)
    plt.annotate(f'({subtitle})', (4.5, ylim[1]-(ylim[1]-ylim[0])/8), fontsize=18)

    if j == 3:
        plt.legend(loc='lower center',ncol=1, fontsize=10) #

plt.subplots_adjust(wspace=0.4, hspace=0.3)