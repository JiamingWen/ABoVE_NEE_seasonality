'''plot scatterplots for seasonality features of simulated NEE and component fluxes'''
'''modified from plot_NEE_component_phenology_bias.py'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

high_model_subset = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0'] # , 'IBIS' #'ISBA-CTRIP', 'LPJ', 'LPX-Bern', 'SDGVM', 'CLASSIC', 'CLM5.0'
low_model_subset = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES'] #, 'YIBs', 'CABLE-POP', 'ISAM'

weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'


# seasonal cycle
# TRENDY
seasonal_df_TRENDYv11NEE = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11GPP = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11GPP_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Ra = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Ra_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Rh= pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Rh_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11Reco = seasonal_df_TRENDYv11Ra + seasonal_df_TRENDYv11Rh

# X-BASE
seasonal_df_NEEobservations = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_NEEobservations_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_GPPobservations = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_GPPobservations_{regionname}_{lcname}_{weightname}.csv")
seasonal_df_TRENDYv11NEE['X-BASE'] = seasonal_df_NEEobservations['FluxCOM-X-NEE']
seasonal_df_TRENDYv11GPP['X-BASE'] = seasonal_df_GPPobservations['FluxCOM-X-GPP']
seasonal_df_TRENDYv11Reco['X-BASE'] = seasonal_df_TRENDYv11NEE['X-BASE'] + seasonal_df_TRENDYv11GPP['X-BASE']
seasonal_df_TRENDYv11NEE['ABCflux'] = seasonal_df_NEEobservations['ABCflux-NEE']
seasonal_df_TRENDYv11GPP['ABCflux'] = seasonal_df_GPPobservations['ABCflux-GPP']
seasonal_df_TRENDYv11Reco['ABCflux'] = seasonal_df_TRENDYv11NEE['ABCflux'] + seasonal_df_TRENDYv11GPP['ABCflux']

component_name = 'Reco' # GPP Reco
if component_name == 'GPP':
    axis_label = 'GPP'
elif component_name == 'Reco':
    axis_label = '$\it{R}_{eco}$ '

# specify data frame
if component_name == 'GPP':
    seasonal_df_TRENDYv11_comp = seasonal_df_TRENDYv11GPP
elif component_name == 'Reco':
    seasonal_df_TRENDYv11_comp = seasonal_df_TRENDYv11Reco

# make adjustment for text lable location
if component_name == 'GPP':
    x_offset1 = np.ones(12)*0.01; x_offset1[2]= -0.02; x_offset1[5]= -0.03; x_offset1[6]= -0.05; x_offset1[9]= -0.09
    y_offset1 = np.ones(12)*0.01; y_offset1[2]= 0.03; y_offset1[5]= 0.03; y_offset1[6]= 0; y_offset1[9]= 0; y_offset1[10]= -0.05
    x_offset2 = np.ones(11)*0.005; x_offset2[4]= -0.04; x_offset2[5]= -0.02; x_offset2[7]= -0.04; x_offset2[8]= -0.05; x_offset2[10]= -0.01
    y_offset2 = np.ones(11)*0.01; y_offset2[4]= 0.04; y_offset2[5]= 0.03; y_offset2[6]= -0.01; y_offset2[7]= 0; y_offset2[8]= 0.02; y_offset2[10]= 0.07
    x_offset3 = np.ones(11)*0.005; x_offset3[0]= -0.02; x_offset3[3]= 0.01; x_offset3[7]= -0.035; x_offset3[8]= -0.02; x_offset3[9]= -0.01; x_offset3[10]= 0.01
    y_offset3 = np.ones(11)*0.01; y_offset3[0]= 0.05; y_offset3[3]= -0.02; y_offset3[6]= -0.02; y_offset3[7]= -0.02; y_offset3[8]= 0.02; y_offset3[9]= 0.08; y_offset3[10]= -0.05
    xlim1 = (0.15, 0.55); ylim1 = (-0.6, 0.65)
    xlim2 = (0.75, 1); ylim2 = (0.4, 1.6)
    xlim3 = (0.3, 0.6); ylim3 = (-0.6, 0.5)
    xticks1 = np.arange(0.2, 0.6, 0.1); yticks1 = np.arange(-0.6, 0.8, 0.3)
    xticks2 = np.arange(0.80, 1.05, 0.05); yticks2 = np.arange(0.4, 1.8, 0.2)
    xticks3 = np.arange(0.3, 0.7, 0.1); yticks3 = np.arange(-0.6, 0.6, 0.2)
elif component_name == 'Reco':
    x_offset1 = np.ones(12)*0.01; x_offset1[0]= -0.08; x_offset1[3]= -0.02; x_offset1[5]= -0.03; x_offset1[6]= -0.05
    y_offset1 = np.ones(12)*0.01; y_offset1[0]= 0.03; y_offset1[3]= 0.03; y_offset1[5]= 0.03; y_offset1[6]= 0; y_offset1[10]= -0.05
    x_offset2 = np.ones(11)*0.005; x_offset2[0]= -0.10; x_offset2[3]= -0.04; x_offset2[4]= -0.02; x_offset2[10]= -0.05
    y_offset2 = np.ones(11)*0.01; y_offset2[0]= -0.03; y_offset2[3]= 0.04; y_offset2[4]= 0.03; y_offset2[8]= -0.04; y_offset2[10]= -0.08
    x_offset3 = np.ones(11)*0.005; x_offset3[0]= 0.02; x_offset3[5] = -0.05
    y_offset3 = np.ones(11)*0.01; y_offset3[0]= 0.02; y_offset3[9]= 0.05; y_offset3[10]= -0.07
    xlim1 = (0.2, 0.6); ylim1 = (-0.6, 0.65)
    xlim2 = (0.55, 1.05); ylim2 = (0.4, 1.6)
    xlim3 = (0.4, 1.2); ylim3 = (-0.6, 0.5)
    xticks1 = np.arange(0.2, 0.6, 0.1); yticks1 = np.arange(-0.6, 0.8, 0.3)
    xticks2 = np.arange(0.6, 1.0, 0.1); yticks2 = np.arange(0.4, 2.0, 0.4)
    xticks3 = np.arange(0.4, 1.2, 0.2); yticks3 = np.arange(-0.6, 0.6, 0.2)

fig, axs = plt.subplots(1, 3, figsize=(16, 4))

# Earlier start of growing season
reference_models = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
bias_models = ['LPX-Bern','JULES', 'VISIT', 'JSBACH', 'SDGVM', 'VISIT-NIES']
EC_model = ['X-BASE', 'ABCflux']
models = reference_models + bias_models + EC_model
colors =  ['#396BB8'] * len(reference_models) + ['#D4631D'] * len(bias_models) + ['#56983F'] * len(EC_model)
scatter_types = ['o'] * len(reference_models) + ['o'] * len(bias_models) + ['d'] * len(EC_model)
scatter_sizes = [70] * len(reference_models) + [70] * len(bias_models) + [80] * len(EC_model)
ax1 = axs[0]
for i, model_name in enumerate(models):
    x = seasonal_df_TRENDYv11_comp.loc[4,model_name] / max(seasonal_df_TRENDYv11_comp.loc[:,model_name])
    y = seasonal_df_TRENDYv11NEE.loc[4,model_name] / min(seasonal_df_TRENDYv11NEE.loc[:,model_name])
    ax1.scatter(x, y, color=colors[i], marker=scatter_types[i], s=scatter_sizes[i])
    ax1.text(x+x_offset1[i], y+y_offset1[i], model_name)

# add other low-correlation models
for i, model_name in enumerate(low_model_subset):
    if model_name not in models:
        x = seasonal_df_TRENDYv11_comp.loc[4,model_name] / max(seasonal_df_TRENDYv11_comp.loc[:,model_name])
        y = seasonal_df_TRENDYv11NEE.loc[4,model_name] / min(seasonal_df_TRENDYv11NEE.loc[:,model_name])
        ax1.scatter(x, y, color='#D4631D', marker='o', s=70, alpha=0.5)

plt.text(0.88, 0.05, '(a)', transform=ax1.transAxes, fontsize=16)
ax1.set_xticks(xticks1)
ax1.set_yticks(yticks1)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.set_xlabel(rf'{axis_label}$_{{May}}$ / {axis_label}$_{{max}}$', fontsize=16)
ax1.set_ylabel(r'NEE$_{{May}}$ / NEE$_{{min}}$', fontsize=16)


# NEE vs GPP bias for specific features
# Peak in June
reference_models = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
bias_models = ['ORCHIDEE', 'VISIT', 'JSBACH', 'SDGVM', 'VISIT-NIES']
EC_model = ['X-BASE', 'ABCflux']
models = reference_models + bias_models + EC_model
colors =  ['#396BB8'] * len(reference_models) + ['#D4631D'] * len(bias_models) + ['#56983F'] * len(EC_model)
scatter_types = ['o'] * len(reference_models) + ['o'] * len(bias_models) + ['d'] * len(EC_model)
scatter_sizes = [70] * len(reference_models) + [70] * len(bias_models) + [80] * len(EC_model)
ax2 = axs[1]
for i, model_name in enumerate(models):
    x = seasonal_df_TRENDYv11_comp.loc[5,model_name] / seasonal_df_TRENDYv11_comp.loc[6,model_name]
    y = seasonal_df_TRENDYv11NEE.loc[5,model_name] / seasonal_df_TRENDYv11NEE.loc[6,model_name]
    ax2.scatter(x, y, color=colors[i], marker=scatter_types[i], s=scatter_sizes[i])
    ax2.text(x+x_offset2[i], y+y_offset2[i], model_name)

# add other low-correlation models
for i, model_name in enumerate(low_model_subset):
    if model_name not in models:
        x = seasonal_df_TRENDYv11_comp.loc[5,model_name] / seasonal_df_TRENDYv11_comp.loc[6,model_name]
        y = seasonal_df_TRENDYv11NEE.loc[5,model_name] / seasonal_df_TRENDYv11NEE.loc[6,model_name]
        ax2.scatter(x, y, color='#D4631D', marker='o', s=70, alpha=0.5)

plt.text(0.88, 0.05, '(b)', transform=ax2.transAxes, fontsize=16)
ax2.set_xticks(xticks2)
ax2.set_yticks(yticks2)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.set_xlabel(rf'{axis_label}$_{{June}}$ / {axis_label}$_{{July}}$', fontsize=16)
ax2.set_ylabel(r'NEE$_{{June}}$ / NEE$_{{July}}$', fontsize=16)


# Later end of growing season
reference_models = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
bias_models = ['LPX-Bern','JULES', 'VISIT', 'OCN', 'VISIT-NIES']
EC_model = ['X-BASE', 'ABCflux']
models = reference_models + bias_models + EC_model
colors =  ['#396BB8'] * len(reference_models) + ['#D4631D'] * len(bias_models) + ['#56983F'] * len(EC_model)
scatter_types = ['o'] * len(reference_models) + ['o'] * len(bias_models) + ['d'] * len(EC_model)
scatter_sizes = [70] * len(reference_models) + [70] * len(bias_models) + [80] * len(EC_model)
ax3 = axs[2]
for i, model_name in enumerate(models):
    x = seasonal_df_TRENDYv11_comp.loc[8,model_name]  / max(seasonal_df_TRENDYv11_comp.loc[:,model_name])
    y = seasonal_df_TRENDYv11NEE.loc[8,model_name] / min(seasonal_df_TRENDYv11NEE.loc[:,model_name])
    ax3.scatter(x, y, color=colors[i], marker=scatter_types[i], s=scatter_sizes[i])
    ax3.text(x+x_offset3[i], y+y_offset3[i], model_name)

# add other low-correlation models
for i, model_name in enumerate(low_model_subset):
    if model_name not in models:
        x = seasonal_df_TRENDYv11_comp.loc[8,model_name]  / max(seasonal_df_TRENDYv11_comp.loc[:,model_name])
        y = seasonal_df_TRENDYv11NEE.loc[8,model_name] / min(seasonal_df_TRENDYv11NEE.loc[:,model_name])
        ax3.scatter(x, y, color='#D4631D', marker='o', s=70, alpha=0.5)

plt.text(0.88, 0.05, '(c)', transform=ax3.transAxes, fontsize=16)
ax3.set_xticks(ax3.get_xticks())
ax3.set_yticks(ax3.get_yticks())
ax3.set_xticks(xticks3)
ax3.set_yticks(yticks3)
ax3.tick_params(axis='x', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)
ax3.set_ylim(xlim3)
ax3.set_ylim(ylim3)
ax3.set_xlabel(rf'{axis_label}$_{{September}}$ / {axis_label}$_{{max}}$', fontsize=16)
ax3.set_ylabel(r'NEE$_{{September}}$ / NEE$_{{min}}$', fontsize=16)


plt.subplots_adjust(wspace=0.3)

# save figure
fig.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig5_{component_name}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/Fig5_{component_name}.pdf', dpi=300, bbox_inches='tight')
plt.show()