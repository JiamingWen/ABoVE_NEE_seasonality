import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.regression.linear_model import OLSResults
import math

varname = 'NEE' # NEE GPP Ra Rh Reco
reference = 'obs' # GroupH obs


lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'

high_model_subset = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0'] # , 'IBIS' #'ISBA-CTRIP', 'LPJ', 'LPX-Bern', 'SDGVM', 'CLASSIC', 'CLM5.0'
low_model_subset = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #, 'VISIT-NIES'   


# standardize with minumum NEE
def scale_minumum (vec):
    return -vec / np.min(vec)

# standardize with maximum and minimum
def scale_maximum_minimum (vec):
    return (vec - np.min(vec)) / (np.max(vec) - np.min(vec))


if varname == 'NEE':
    scale_fun = scale_minumum
    ylim = [-1.2,1.2]
    yticks = np.arange(-1, 1.2, 0.5)
    filestr = ''
    ylabel = varname

else:
    scale_fun = scale_maximum_minimum
    ylim = [0,1.2]
    yticks = np.arange(0, 1.2, 0.3)
    filestr = varname
    if varname == 'Ra':
        ylabel = '$\it{R}_{a}$'
    elif varname == 'Rh':
        ylabel = '$\it{R}_{h}$'
    elif varname == 'Reco':
        ylabel = '$\it{R}_{eco}$'
    else:
        ylabel = varname

'''stat'''
# TRENDY
fitting_df_TRENDYv11_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_TRENDYv11_only_seasonal.csv')
# Upscaled EC
fitting_df_NEEobservations_unscaled_only_seasonal = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_NEEobservations_only_seasonal.csv')
fitting_df_NEEobservations_unscaled_only_seasonal.loc[fitting_df_NEEobservations_unscaled_only_seasonal['model_name'] == 'FluxCOM-X-NEE','model_name'] = 'X-BASE'
fitting_df_NEEobservations_unscaled_only_seasonal.loc[fitting_df_NEEobservations_unscaled_only_seasonal['model_name'] == 'ABCflux-NEE','model_name'] = 'ABCflux'
fitting_df = pd.concat((fitting_df_TRENDYv11_unscaled_only_seasonal, fitting_df_NEEobservations_unscaled_only_seasonal))

fitting_df = fitting_df.sort_values('cor', ascending=False)
fitting_df.loc[fitting_df['model_name'].isin(high_model_subset),'color'] = '#396BB8'
fitting_df.loc[fitting_df['model_name'].isin(low_model_subset),'color'] = '#D4631D'
fitting_df.loc[fitting_df['model_name'].isin(['IBIS']),'color'] = 'grey'
fitting_df.loc[fitting_df['model_name'].isin(['X-BASE', 'ABCflux']),'color'] = '#56983F'


'''seasonal cycle'''
# TRENDY
if varname in ['NEE', 'GPP', 'Ra', 'Rh']:
    seasonal_df_TRENDYv11 = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_TRENDYv11{filestr}_{regionname}_{lcname}_{weightname}.csv")
elif varname == 'Reco':
    seasonal_df_TRENDYv11Ra = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_TRENDYv11Ra_{regionname}_{lcname}_{weightname}.csv")
    seasonal_df_TRENDYv11Rh = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_TRENDYv11Rh_{regionname}_{lcname}_{weightname}.csv")
    seasonal_df_TRENDYv11 = seasonal_df_TRENDYv11Ra + seasonal_df_TRENDYv11Rh

# Upscaled EC
seasonal_NEEobservations = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_NEEobservations_{regionname}_{lcname}_{weightname}.csv")
seasonal_NEEobservations.rename(columns={'FluxCOM-X-NEE': 'X-BASE', 'ABCflux-NEE': 'ABCflux'}, inplace=True)
seasonal_GPPobservations = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_GPPobservations_{regionname}_{lcname}_{weightname}.csv")
seasonal_GPPobservations.rename(columns={'FluxCOM-X-GPP': 'X-BASE', 'ABCflux-GPP': 'ABCflux'}, inplace=True)

if varname == 'NEE':
    seasonal_df_EC = seasonal_NEEobservations
elif varname == 'GPP':
    seasonal_df_EC = seasonal_GPPobservations
elif varname == 'Reco':
    seasonal_df_EC = seasonal_NEEobservations + seasonal_GPPobservations
elif varname in ['Ra', 'Rh']:
    seasonal_df_EC = pd.DataFrame(np.nan * np.ones((12, 2)), columns=['X-BASE', 'ABCflux'])

seasonal_df = pd.concat((seasonal_df_TRENDYv11, seasonal_df_EC), axis=1)
seasonal_df = seasonal_df.apply(scale_fun, axis=0)

# reference
# inversions
seasonal_df_inversions = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_inversionsNEE_{regionname}_{lcname}_{weightname}.csv")
inversion_names = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE'] # excluding models without CARVE coverage
seasonal_df_subset_inversion = seasonal_df_inversions[inversion_names]

# GOME-2 SIF
seasonal_df_RS = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal/seasonal_reference_{regionname}_{lcname}_{weightname}.csv")

if reference == 'GroupH': # high-cor TBMs
    seasonal_df_reference = seasonal_df_TRENDYv11[high_model_subset]
    reference_label = 'Higher-correlation TBMs'
    
elif reference == 'obs':
    if varname == 'NEE':
        seasonal_df_reference = seasonal_df_subset_inversion
        reference_label = 'Inversions'
    elif varname == 'GPP':
        seasonal_df_reference = pd.concat((seasonal_GPPobservations['GOSIF-GPP'], seasonal_df_RS['GOME2_SIF']), axis=1)
        reference_label = 'GOME-2 SIF & GOSIF GPP'
        seasonal_df_APAR = seasonal_df_RS['APAR']
    elif varname == 'Reco':
        seasonal_df_reference = seasonal_df_subset_inversion
        for model_name in seasonal_df_subset_inversion.columns:
            seasonal_df_reference[model_name] = seasonal_df_subset_inversion[model_name] + seasonal_GPPobservations['GOSIF-GPP']
        reference_label = 'Inversions + GOSIF GPP'

seasonal_df_reference = seasonal_df_reference.apply(scale_fun, axis=0)
seasonal_df_reference_mean = seasonal_df_reference.mean(axis=1)
seasonal_df_reference_std = seasonal_df_reference.std(axis=1)   


'''plot'''
num_models = fitting_df.shape[0]
fig, ax = plt.subplots((num_models + 1) // 3, 3, figsize=(9, 16), sharex='col', sharey='row')
subplot_id = 0

mean_seasonal_NEE_diff_df = pd.DataFrame()
for i in np.arange(num_models):

    model_name = fitting_df.iloc[i]['model_name']
    color = fitting_df.iloc[i]['color']

    subplot_id += 1
    ax1 = plt.subplot((num_models + 2) // 3, 3, subplot_id)

    # reference
    ax1.fill_between(np.arange(4,12), seasonal_df_reference_mean[3:11]-seasonal_df_reference_std[3:11], seasonal_df_reference_mean[3:11]+seasonal_df_reference_std[3:11], alpha=0.2,color='black') #,edgecolor='none'

    if varname == 'GPP' and reference == 'obs': # plot APAR
        plt.scatter(np.arange(1,13),scale_fun(seasonal_df_APAR), marker='+',color=['purple'], s=80)

    # individual model
    ax1.plot(np.arange(4,12),seasonal_df[model_name][3:11], linestyle='-',color=color, linewidth=2)
    
    mean_seasonal_NEE_diff = np.mean(abs(seasonal_df_reference_mean[3:11]-seasonal_df[model_name][3:11])) # only growing season
    cor = fitting_df.iloc[i]['cor']
    subtitle = chr(ord('`')+i+1)
    ax1.annotate(f'({subtitle}) {model_name}', (4, ylim[1]-(ylim[1]-ylim[0])/8), fontsize=18)
    # ax1.annotate(f'Cor={"{:.2f}".format(cor)}', (8, ylim[0]+(ylim[1]-ylim[0])/6), fontsize=14)
    ax1.annotate(f'MAD={"{:.2f}".format(mean_seasonal_NEE_diff)}', (7.5, ylim[0]+(ylim[1]-ylim[0])/14), fontsize=14)

    ax1.set_xlim(3.5,11.5) #4,11
    ax1.set_ylim(ylim)
    ax1.set_xticks(np.arange(4,12))
    ax1.set_yticks(yticks)
    ax1.tick_params(axis='both', which='major', labelsize=16, direction='in')

    if subplot_id >=15:
        ax1.xaxis.set_tick_params(labelbottom=True)

    mean_seasonal_NEE_diff_df = pd.concat((mean_seasonal_NEE_diff_df, pd.DataFrame([[model_name, cor, mean_seasonal_NEE_diff]], columns=['model_name', 'cor', 'mean_seasonal_diff'])))
mean_seasonal_NEE_diff_df.to_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/seasonal_diff/seasonal_mad_{reference}_{varname}_{regionname}{lc_filestr}_{weightname}.csv', index=False)


# Set shared labels
fig.text(0.5, 0.07, 'Month', ha='center', fontsize=22)
fig.text(0.02, 0.5, f'Rescaled {ylabel}', va='center', rotation='vertical', fontsize=22)

# Add legend
handles = [
    plt.Line2D([0], [0], color='#396BB8', lw=2, label='Higher-correlation TBM'),
    plt.Line2D([0], [0], color='#D4631D', lw=2, label='Lower-correlation TBM'),
    plt.Line2D([0], [0], color='#56983F', lw=2, label='Upscaled EC'),
    plt.Polygon([[0, 0]], closed=True, fill=True, color='black', alpha=0.2, label=reference_label)
]
if varname in ['Ra', 'Rh']:
    handles = [handles[i] for i in [0, 1, 3]]

if varname == 'GPP' and reference == 'obs':
    handles.append(plt.Line2D([0], [0], color='purple', marker='+', markersize=10, linestyle='None', label='APAR'))

fig.legend(handles=handles, loc='lower center', fontsize=14, bbox_to_anchor=(0.5, 0.015), ncol=math.ceil(len(handles)/2))

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/figures/Fig3_{varname}_{reference}{lc_filestr}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/result/figures/Fig3_{varname}_{reference}{lc_filestr}.pdf', dpi=300, bbox_inches='tight')
