# plot figures for ABoVE STM 2024

##########################################################
# Fig 1 for Carbon Dynamics WG: Correlation against Arctic-Cap measurements

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
inversion_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']

fitting_df1 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats.csv')
fitting_df1 = fitting_df1[fitting_df1.model_name.isin(model_names)]
fitting_df2 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats_inversion.csv')

stat_var = 'cor'; xlim = [-0.2, 1]
fitting_df1_sorted = fitting_df1.sort_values(stat_var)
fitting_df2_sorted = fitting_df2.sort_values(stat_var)

fitting_df1_sorted_low = fitting_df1_sorted[fitting_df1_sorted['cor']<0.6]
fitting_df1_sorted_mid = fitting_df1_sorted[(fitting_df1_sorted['cor']>0.6) & (fitting_df1_sorted['cor']<0.7)]
fitting_df1_sorted_high = fitting_df1_sorted[fitting_df1_sorted['cor']>0.7]

fig, ax = plt.subplots(figsize=(6,10))
plt.scatter(fitting_df1_sorted_low[stat_var], fitting_df1_sorted_low['model_name'], marker='s', color='#e57f3f')
plt.scatter(fitting_df1_sorted_mid[stat_var], fitting_df1_sorted_mid['model_name'], marker='s', color='#6fb557')
plt.scatter(fitting_df1_sorted_high[stat_var], fitting_df1_sorted_high['model_name'], marker='s', color='#5986cb')
plt.scatter(fitting_df2_sorted[stat_var], fitting_df2_sorted['model_name'], marker='D', color='#000000')

colors = ['#e57f3f'] * fitting_df1_sorted_low.shape[0] + ['#6fb557'] * fitting_df1_sorted_mid.shape[0] + ['#5986cb'] * fitting_df1_sorted_high.shape[0] + ['#000000'] * fitting_df2_sorted.shape[0]
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)

plt.xlim(xlim)
plt.ylim(-1, 30)
plt.xlabel('Correlation with Arctic-CAP measurements', fontsize=15)
plt.xticks(fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)

plt.axhline(y = 15.5, color = 'r', linestyle = '--') 
plt.annotate("Inversions", (-0.13, 29), fontsize=15)
plt.annotate("TRENDY v11", (-0.13, 14.5), fontsize=15)
plt.show()


##########################################################
# Fig 2 for Carbon Dynamics WG: standardized seasonal cycle
colors = ['#000000', '#5986cb', '#6fb557', '#e57f3f']
fitting_df_list = [fitting_df2_sorted, fitting_df1_sorted_high, fitting_df1_sorted_mid, fitting_df1_sorted_low]
# seasonal_df_TRENDY = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/TRENDY/ABoVE_2017_TRENDY_nee_seasonal.csv')
seasonal_df_TRENDY = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/regression/ABoVE_2017_TRENDY_seasonal_all.csv')
seasonal_df_TRENDY = seasonal_df_TRENDY[model_names]
seasonal_df_inversion = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/inversion/ABoVE_2017_inversion_seasonal.csv')
seasonal_df = pd.concat((seasonal_df_TRENDY, seasonal_df_inversion), axis=1)


# standardize with maximum
def scale_maximum (vec):
    return vec / np.max(abs(vec))
seasonal_df = seasonal_df.apply(scale_maximum, axis=0)

# fig = plt.figure(figsize=(4,8))
fig, ax = plt.subplots(figsize=(4,10), nrows=4, sharex=True)
for i in np.arange(len(colors)):
    ax1 = plt.subplot(4,1,i+1)
    color = colors[i]
    fitting_df_subset = fitting_df_list[i]

    modellist = fitting_df_subset['model_name'].tolist()
    seasonal_df_subset = seasonal_df.loc[:,modellist]
    seasonal_mean_ensemble = seasonal_df_subset.mean(axis=1)

    plt.plot(np.arange(4,12),seasonal_mean_ensemble, linestyle='-',color=color, linewidth=5)
    plt.ylim(-1.2,1)
    ax1.set_xticks(np.arange(4,12))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    ax1.tick_params(axis ='x', length = 7, direction ='in')
    for model_name in modellist:
        plt.plot(np.arange(4,12),seasonal_df.loc[:,model_name], linestyle='-',color=color, linewidth=.5)

plt.xlabel('Month in 2017', fontsize=15)
plt.text(2, 3, 'Standardized seasonal NEE', va='center', fontsize=15, rotation='vertical')
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

# ##########################################################
# # Fig 1 for Carbon Dynamics WG: Correlation against Arctic-Cap measurements

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# model_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
# inversion_names = ['CAMS', 'CT2022', 'NISMON']

# fitting_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats.csv')

# fitting_df1 = fitting_df[fitting_df.model_name.isin(model_names)]
# fitting_df2 = fitting_df[fitting_df.model_name.isin(inversion_names)]

# stat_var = 'cor'; xlim = [-0.2, 1]
# fitting_df1_sorted = fitting_df1.sort_values(stat_var)
# fitting_df2_sorted = fitting_df2.sort_values(stat_var)

# fitting_df1_sorted_low = fitting_df1_sorted[fitting_df1_sorted['cor']<0.6]
# fitting_df1_sorted_mid = fitting_df1_sorted[(fitting_df1_sorted['cor']>0.6) & (fitting_df1_sorted['cor']<0.7)]
# fitting_df1_sorted_high = fitting_df1_sorted[fitting_df1_sorted['cor']>0.7]

# fig, ax = plt.subplots(figsize=(6,7))
# plt.scatter(fitting_df1_sorted_low[stat_var], fitting_df1_sorted_low['model_name'], marker='s', color='#e57f3f')
# plt.scatter(fitting_df1_sorted_mid[stat_var], fitting_df1_sorted_mid['model_name'], marker='s', color='#6fb557')
# plt.scatter(fitting_df1_sorted_high[stat_var], fitting_df1_sorted_high['model_name'], marker='s', color='#5986cb')
# plt.scatter(fitting_df2_sorted[stat_var], fitting_df2_sorted['model_name'], marker='D', color='#000000')

# colors = ['#e57f3f'] * fitting_df1_sorted_low.shape[0] + ['#6fb557'] * fitting_df1_sorted_mid.shape[0] + ['#5986cb'] * fitting_df1_sorted_high.shape[0] + ['#000000'] * fitting_df2_sorted.shape[0]
# for ytick, color in zip(ax.get_yticklabels(), colors):
#     ytick.set_color(color)

# plt.xlim(xlim)
# plt.xlabel('Correlation with Arctic-CAP measurements', fontsize=15)
# plt.xticks(fontsize=15) #np.arange(-0.2, 1, 0.2), 
# plt.yticks(fontsize=15)

# plt.axhline(y = 15.5, color = 'r', linestyle = '--') 
# plt.annotate("Inversions", (-0.13, 18), fontsize=15)
# plt.annotate("TRENDY v11", (-0.13, 14.5), fontsize=15)
# plt.show()


# ##########################################################
# # Fig 2 for Carbon Dynamics WG: standardized seasonal cycle
# colors = ['#000000', '#5986cb', '#6fb557', '#e57f3f']
# fitting_df_list = [fitting_df2_sorted, fitting_df1_sorted_high, fitting_df1_sorted_mid, fitting_df1_sorted_low]
# seasonal_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/regression/ABoVE_2017_TRENDY_seasonal_all.csv')

# # standardize with maximum
# def scale_maximum (vec):
#     return vec / np.max(abs(vec))
# seasonal_df = seasonal_df.apply(scale_maximum, axis=0)

# # fig = plt.figure(figsize=(4,8))
# fig, ax = plt.subplots(figsize=(4,10), nrows=4, sharex=True)
# for i in np.arange(len(colors)):
#     ax1 = plt.subplot(4,1,i+1)
#     color = colors[i]
#     fitting_df_subset = fitting_df_list[i]

#     modellist = fitting_df_subset['model_name'].tolist()
#     seasonal_df_subset = seasonal_df.loc[:,modellist]
#     seasonal_mean_ensemble = seasonal_df_subset.mean(axis=1)

#     plt.plot(np.arange(4,12),seasonal_mean_ensemble, linestyle='-',color=color, linewidth=5)
#     plt.ylim(-1.2,1)
#     ax1.set_xticks(np.arange(4,12))
#     plt.xticks(fontsize=15)
#     plt.yticks(fontsize=12)
#     ax1.tick_params(axis ='x', length = 7, direction ='in')
#     for model_name in modellist:
#         plt.plot(np.arange(4,12),seasonal_df.loc[:,model_name], linestyle='-',color=color, linewidth=.5)

# plt.xlabel('Month in 2017', fontsize=15)
# plt.text(2, 3, 'Standardized seasonal NEE', va='center', fontsize=15, rotation='vertical')
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()


##########################################################
# Fig 1 for Modeling WG: v9 vs v11 Correlation with Arctic-CAP
fitting_df_v9 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats_TRENDYv9.csv')
fitting_df_v11 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats.csv')
model_names_v11 = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
fitting_df_v11 = fitting_df_v11[fitting_df_v11['model_name'].isin(model_names_v11)]
fitting_df = pd.merge(fitting_df_v11, fitting_df_v9, on='model_name', how='outer', suffixes=('_v11', '_v9'))

fitting_df = fitting_df.sort_values('cor_v11')

stat_var = 'cor'; xlim = [-0.2, 1]

fig = plt.figure(figsize=(6,7))
plt.scatter(fitting_df[f'{stat_var}_v9'], fitting_df['model_name'], marker='s', s=50, color='red', label='TRENDY v9')
plt.scatter(fitting_df[f'{stat_var}_v11'], fitting_df['model_name'], marker='^', s=50, color='black', label='TRENDY v11')

plt.xlim(xlim)
plt.xlabel('Correlation with Arctic-CAP measurements', fontsize=17)
plt.xticks(fontsize=16) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=16)
plt.legend(loc='center left', fontsize=15)
plt.show()

##########################################################
# Fig 2 for Modeling WG: v9 vs v11 Seasonal variations
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd

var_name = 'nee'
model_names_v9 = ['CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT']
seasonal_df_TRENDYv9 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/TRENDY/ABoVE_2017_TRENDY_{var_name}_seasonal_TRENDYv9.csv')
seasonal_df_TRENDYv11 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/TRENDY/ABoVE_2017_TRENDY_{var_name}_seasonal_all.csv')


fig = plt.figure(figsize=(18,12))
subplot_id = 0
ylim1 = -2.5
ylim2 = 2.5

for model_name in model_names_v9:
    subplot_id += 1
    
    ax = plt.subplot(3,4,subplot_id)
    plt.plot(np.arange(4,12),seasonal_df_TRENDYv9[model_name], linestyle='-',color='red')
    plt.plot(np.arange(4,12),seasonal_df_TRENDYv11[model_name], linestyle='-',color='black')
    
    ax.set_xticks(np.arange(4,12))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.tick_params(axis ='x', length = 7, direction ='in')

    plt.title(model_name, fontsize=20)
    plt.ylim(ylim1, ylim2)

fig.text(0.07, 0.5, f'{var_name.upper()} ' + '($\mu$mol m$^{-2}$ s$^{-1}$)', va='center', fontsize=25, rotation='vertical')
fig.text(0.5, 0.05, 'Month in 2017', ha='center', fontsize=25)

plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()


##############################################################
# Fig 1 for Michalak TE 2021 project slides
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
stat_var = 'cor'; xlim = [-0.2, 1]

TRENDYv11_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
reference_names = ['APAR', 'PAR', 'FPAR', 'LAI']

fitting_df0 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats.csv')
fitting_df_TRENDY = fitting_df0[fitting_df0['model_name'].isin(TRENDYv11_names)]
fitting_df_RS = fitting_df0[fitting_df0['model_name'].isin(reference_names)]
fitting_df_inversions = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats_inversion.csv')

fitting_df_TRENDY_sorted = fitting_df_TRENDY.sort_values(stat_var)
fitting_df_RS_sorted = fitting_df_RS.sort_values(stat_var)
fitting_df_inversions_sorted = fitting_df_inversions.sort_values(stat_var)

fitting_df = pd.concat([fitting_df_TRENDY_sorted, fitting_df_inversions_sorted, fitting_df_RS_sorted], axis=0)

fitting_df_v9 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats_TRENDYv9.csv')
fitting_df = pd.merge(fitting_df, fitting_df_v9, on='model_name', how='outer', suffixes=('_v11', '_v9'))


fig = plt.figure(figsize=(6,10))
plt.scatter(fitting_df['cor_v11'], fitting_df['model_name'], marker='s', color='black', label='TRENDY v11')
# plt.scatter(fitting_df['cor_v9'], fitting_df['model_name'], marker='^', s=50, color='red', label='TRENDY v9') # comment this line to plot v9 only
plt.axhline(y = 15.5, color = 'grey', linestyle = '--')
plt.axhline(y = 29.5, color = 'grey', linestyle = '--')
# plt.axhline(y = 33.5, color = 'r', linestyle = '--')
plt.xlim(xlim)
plt.ylim(-1, fitting_df.shape[0])
plt.xlabel('Correlation with Arctic-CAP measurements', fontsize=17)
plt.xticks(fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)

plt.annotate("Remote Sensing", (-0.13, 32.5), fontsize=15)
plt.annotate("Inversions", (-0.13, 28), fontsize=15)
plt.annotate("TRENDY", (-0.13, 14), fontsize=15)

# plt.legend(loc=(0.1, 0.3),fontsize=15)  # comment this line to plot v9 only

plt.show()


##############################################################
# Fig 1 for Michalak TE 2021 project slides - old version 2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# fitting_df_v9 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats_TRENDYv9.csv')
# fitting_df_v11 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats.csv')

# model_names_v11 = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
# inversion_names = ['CAMS', 'CT2022', 'NISMON']
# reference_names = ['APAR', 'PAR', 'FPAR', 'LAI']

# fitting_df_v11 = fitting_df_v11[fitting_df_v11['model_name'].isin(model_names_v11 + inversion_names + reference_names)]
# fitting_df = pd.merge(fitting_df_v11, fitting_df_v9, on='model_name', how='outer', suffixes=('_v11', '_v9'))

# fitting_df1 = fitting_df[fitting_df.model_name.isin(model_names_v11)]
# fitting_df2 = fitting_df[fitting_df.model_name.isin(inversion_names)]
# fitting_df3 = fitting_df[fitting_df.model_name.isin(reference_names)]

# fitting_df1_sorted = fitting_df1.sort_values('cor_v11')
# fitting_df2_sorted = fitting_df2.sort_values('cor_v11')
# fitting_df3_sorted = fitting_df3.sort_values('cor_v11')

# fitting_df = pd.concat([fitting_df1_sorted, fitting_df2_sorted, fitting_df3_sorted], axis=0)

# xlim = [-0.2, 1]

# fig = plt.figure(figsize=(6,9))
# plt.scatter(fitting_df['cor_v11'], fitting_df['model_name'], marker='s', s=50, color='black', label='TRENDY v11')
# # plt.scatter(fitting_df['cor_v9'], fitting_df['model_name'], marker='^', s=50, color='red', label='TRENDY v9') # comment this line to plot v9 only
# plt.axhline(y = 15.5, color = 'r', linestyle = '--') 
# plt.axhline(y = 18.5, color = 'r', linestyle = '--')

# plt.xlim(xlim)
# plt.ylim(-1,23)
# plt.xlabel('Correlation with Arctic-CAP measurements', fontsize=17)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# # plt.legend(loc='center left', fontsize=15)  # comment this line to plot v9 only

# plt.annotate("Remote Sensing", (-0.13, 22), fontsize=16)
# plt.annotate("Inversions", (-0.13, 17.5), fontsize=16)
# plt.annotate("TRENDY", (-0.13, 14.5), fontsize=16)

# plt.show()

# ##############################################################
# # Fig 1 for Michalak TE 2021 project slides - old version
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# fitting_df_v9 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats_TRENDYv9.csv')
# fitting_df_v11 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/arctic_cap_airborne/ABoVE_2017_arctic_cap_airborne_original_stats.csv')

# model_names_v11 = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
# inversion_names = ['CAMS', 'CT2022', 'NISMON']
# reference_names = ['APAR', 'PAR', 'FPAR', 'LAI']

# fitting_df_others = fitting_df_v11[fitting_df_v11['model_name'].isin(inversion_names + reference_names)]
# fitting_df_v11 = fitting_df_v11[fitting_df_v11['model_name'].isin(model_names_v11)]
# fitting_df_v9 = pd.concat([fitting_df_v9, fitting_df_others], axis=0)
# fitting_df = pd.merge(fitting_df_v11, fitting_df_v9, on='model_name', how='outer', suffixes=('_v11', '_v9'))

# fitting_df1 = fitting_df[fitting_df.model_name.isin(model_names_v11)]
# fitting_df2 = fitting_df[fitting_df.model_name.isin(inversion_names)]
# fitting_df3 = fitting_df[fitting_df.model_name.isin(reference_names)]

# fitting_df1a = fitting_df1[~fitting_df1.cor_v9.isnull()]
# fitting_df1a_sorted = fitting_df1a.sort_values('cor_v9') #, na_position='first'
# fitting_df1b = fitting_df1[fitting_df1.cor_v9.isnull()] # nan values for v9, rank by v11 cor
# fitting_df1b_sorted = fitting_df1b.sort_values('cor_v11')
# fitting_df1_sorted = pd.concat([fitting_df1b_sorted, fitting_df1a_sorted], axis=0)

# fitting_df2_sorted = fitting_df2.sort_values('cor_v9')
# fitting_df3_sorted = fitting_df3.sort_values('cor_v9')

# fitting_df = pd.concat([fitting_df1_sorted, fitting_df2_sorted, fitting_df3_sorted], axis=0)

# xlim = [-0.2, 1]

# fig = plt.figure(figsize=(6,9))
# plt.scatter(fitting_df['cor_v9'], fitting_df['model_name'], marker='s', s=50, color='black', label='TRENDY v9')
# # plt.scatter(fitting_df['cor_v11'], fitting_df['model_name'], marker='^', s=50, color='red', label='TRENDY v11') # comment this line to plot v9 only
# plt.axhline(y = 15.5, color = 'r', linestyle = '--') 
# plt.axhline(y = 18.5, color = 'r', linestyle = '--')

# plt.xlim(xlim)
# plt.ylim(-1,23)
# plt.xlabel('Correlation with Arctic-CAP measurements', fontsize=17)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# # plt.legend(loc='center left', fontsize=15)  # comment this line to plot v9 only

# plt.annotate("Remote Sensing", (-0.13, 22), fontsize=16)
# plt.annotate("Inversions", (-0.13, 17.5), fontsize=16)
# plt.annotate("TRENDY", (-0.13, 14.5), fontsize=16)

# plt.show()