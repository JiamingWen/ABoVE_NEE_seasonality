'''plot histgrams of model performance'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''figure 1'''
lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

stat_var = 'cor'; xlim = [-0.1, 0.75]

# # evaluation on the original data (without linear regression)
# fitting_df_TRENDYv11 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}.csv')
# fitting_df_inversions = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}.csv')
# fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
# fitting_df_inversions.loc[fitting_df_inversions['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
# fitting_df_NEEobservations = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_NEEobservations{lc_filestr}.csv')
# fitting_df_NEEobservations.loc[fitting_df_NEEobservations['model_name'] == 'FluxCOM-X-NEE','model_name'] = 'X-BASE'
# fitting_df_NEEobservations.loc[fitting_df_NEEobservations['model_name'] == 'ABCflux-NEE','model_name'] = 'ABCflux'

# evaluation only on the mean seasonal cycles
fitting_df_TRENDYv11 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_only_seasonal.csv')
# fitting_df_TRENDYv11 = fitting_df_TRENDYv11.loc[~fitting_df_TRENDYv11['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_only_seasonal.csv')
fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions.loc[fitting_df_inversions['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_NEEobservations = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_NEEobservations{lc_filestr}_only_seasonal.csv')
fitting_df_NEEobservations.loc[fitting_df_NEEobservations['model_name'] == 'FluxCOM-X-NEE','model_name'] = 'X-BASE'


# Combine dataframes into a single dataframe with a 'group' column
fitting_df_TRENDYv11['group'] = 'TRENDY TBMs'
fitting_df_inversions['group'] = 'Inversions'
fitting_df_NEEobservations['group'] = 'Upscaled fluxes'
combined_data = pd.concat([fitting_df_TRENDYv11, fitting_df_inversions, fitting_df_NEEobservations])

# Plot boxplots and overlay individual points
fig, ax = plt.subplots(figsize=(6, 3))
# Define colors for groups and subgroups
colors = {'Inversions': 'black', 'Upscaled fluxes': '#56983f'}

# Assign colors to TRENDY TBMs based on skill level
high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM']

fitting_df_TRENDYv11['color'] = '#396bb8'  # Default color for high skill
fitting_df_TRENDYv11.loc[fitting_df_TRENDYv11['model_name'].isin(low_skill_TRENDY), 'color'] = '#d4631d'

# Add TRENDY TBMs colors to the main color dictionary
colors.update({'TRENDY TBMs': fitting_df_TRENDYv11['color']})

# Define the add_jitter function
def add_jitter(values, jitter_amount=0.1, seed=None):
    """Add random jitter to values with an optional random seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
    return values + np.random.uniform(-jitter_amount, jitter_amount, size=len(values))

# Overlay individual points
for group, group_data in combined_data.groupby('group'):
    jitter_x = group_data[stat_var]
    jitter_y = group_data['group'].map({'TRENDY TBMs': 1, 'Upscaled fluxes': 2, 'Inversions': 3})
    if group == 'Upscaled fluxes':
        jitter_y = add_jitter(jitter_y, jitter_amount=0)
    else:
        jitter_y = add_jitter(jitter_y, jitter_amount=0.4, seed=5)
    ax.scatter(jitter_x, jitter_y, alpha=0.7, color=colors[group])

# Customize plot
ax.set_xticks(np.arange(xlim[0] + 0.1, xlim[1] + 0.1, 0.1))
ax.set_xticklabels([f'{tick:.1f}' for tick in np.arange(xlim[0] + 0.1, xlim[1] + 0.1, 0.1)], fontsize=12)
ax.set_xlim(xlim)
ax.set_ylim(0.5, 3.5)
ax.set_yticks([1, 2, 3])
ax.tick_params(axis='y', length=0)  # Remove tick symbols
ax.set_yticklabels(['TRENDY\n  TBMs', 'Upscaled\n  fluxes', 'Atmospheric\n  Inversions'], fontsize=12, ha='center')
ax.set_xlabel(r'Correlation with CO$_{2}$ observations', fontsize=14)
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Remove box boundary
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

plt.suptitle('')  # Remove automatic title added by pandas boxplot
plt.tight_layout()
plt.show()


'''figure 2: transpose of figure 1'''
lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

stat_var = 'cor'; ylim = [-0.1, 0.75]

# # evaluation on the original data (without linear regression)
# fitting_df_TRENDYv11 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}.csv')
# fitting_df_inversions = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}.csv')
# fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
# fitting_df_inversions.loc[fitting_df_inversions['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
# fitting_df_NEEobservations = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_NEEobservations{lc_filestr}.csv')
# fitting_df_NEEobservations.loc[fitting_df_NEEobservations['model_name'] == 'FluxCOM-X-NEE','model_name'] = 'X-BASE'
# fitting_df_NEEobservations.loc[fitting_df_NEEobservations['model_name'] == 'ABCflux-NEE','model_name'] = 'ABCflux'

# evaluation only on the mean seasonal cycles
fitting_df_TRENDYv11 = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_only_seasonal.csv')
# fitting_df_TRENDYv11 = fitting_df_TRENDYv11.loc[~fitting_df_TRENDYv11['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
fitting_df_inversions = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_only_seasonal.csv')
fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
fitting_df_inversions.loc[fitting_df_inversions['model_name'] == 'MIROC','model_name'] = 'MIROC4-ACTM'
fitting_df_NEEobservations = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_NEEobservations{lc_filestr}_only_seasonal.csv')
fitting_df_NEEobservations.loc[fitting_df_NEEobservations['model_name'] == 'FluxCOM-X-NEE','model_name'] = 'X-BASE'


# Combine dataframes into a single dataframe with a 'group' column
fitting_df_TRENDYv11['group'] = 'TRENDY TBMs'
fitting_df_inversions['group'] = 'Inversions'
fitting_df_NEEobservations['group'] = 'Upscaled fluxes'
combined_data = pd.concat([fitting_df_TRENDYv11, fitting_df_inversions, fitting_df_NEEobservations])

# Plot boxplots and overlay individual points
fig, ax = plt.subplots(figsize=(4, 3.5))
# Define colors for groups and subgroups
colors = {'Inversions': 'black', 'Upscaled fluxes': '#56983f'}

# Assign colors to TRENDY TBMs based on skill level
high_skill_TRENDY = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_skill_TRENDY = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM']

fitting_df_TRENDYv11['color'] = '#396bb8'  # Default color for high skill
fitting_df_TRENDYv11.loc[fitting_df_TRENDYv11['model_name'].isin(low_skill_TRENDY), 'color'] = '#d4631d'

# Add TRENDY TBMs colors to the main color dictionary
colors.update({'TRENDY TBMs': fitting_df_TRENDYv11['color']})

# Define the add_jitter function
def add_jitter(values, jitter_amount=0.1, seed=None):
    """Add random jitter to values with an optional random seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
    return values + np.random.uniform(-jitter_amount, jitter_amount, size=len(values))

# Overlay individual points
for group, group_data in combined_data.groupby('group'):
    jitter_x = group_data[stat_var]
    jitter_y = group_data['group'].map({'TRENDY TBMs': 1, 'Upscaled fluxes': 2, 'Inversions': 3})
    if group == 'Upscaled fluxes':
        jitter_y = add_jitter(jitter_y, jitter_amount=0)
    else:
        jitter_y = add_jitter(jitter_y, jitter_amount=0.5, seed=5)
    ax.scatter(jitter_y, jitter_x, alpha=0.7, color=colors[group])

# Customize plot
ax.set_yticks(np.arange(ylim[0] + 0.1, ylim[1] + 0.1, 0.1))
ax.set_yticklabels([f'{tick:.1f}' for tick in np.arange(ylim[0] + 0.1, ylim[1] + 0.1, 0.1)], fontsize=12)
ax.set_ylim(ylim)
ax.set_xticks([1, 2, 3])
ax.tick_params(axis='y', length=0)  # Remove tick symbols
ax.set_xticklabels(['TRENDY\n  TBMs', 'Upscaled\n  fluxes', 'Atmospheric\n  Inversions'], fontsize=12, ha='center')
ax.set_ylabel(r'Correlation with CO$_{2}$ observations', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Remove box boundary
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

plt.suptitle('')  # Remove automatic title added by pandas boxplot
plt.tight_layout()
plt.show()



'''figure 3: grouped NEE seasonality'''
varname = 'NEE' # NEE GPP Ra Rh Reco

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

weightname = 'unweighted' #unweighted weighted
regionname = 'ABoVEcore'

high_model_subset = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0', 'IBIS'] # , 'IBIS' #'ISBA-CTRIP', 'LPJ', 'LPX-Bern', 'SDGVM', 'CLASSIC', 'CLM5.0'
low_model_subset = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM'] #, 'VISIT-NIES'   


# standardize with minumum NEE
def scale_minumum (vec):
    return -vec / np.min(vec)

scale_fun = scale_minumum
ylim = [-1.2,1.2]
yticks = np.arange(-1, 1.2, 0.5)
filestr = ''
ylabel = varname

seasonal_df_TRENDYv11 = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11{filestr}_{regionname}_{lcname}_{weightname}.csv")
seasonal_NEEobservations = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_NEEobservations_{regionname}_{lcname}_{weightname}.csv")
seasonal_NEEobservations.rename(columns={'FluxCOM-X-NEE': 'X-BASE', 'ABCflux-NEE': 'ABCflux'}, inplace=True)
seasonal_df_inversions = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_inversionsNEE_{regionname}_{lcname}_{weightname}.csv")
inversion_names = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE'] # excluding models without CARVE coverage
seasonal_df_subset_inversion = seasonal_df_inversions[inversion_names]

seasonal_df = pd.concat((seasonal_df_TRENDYv11, seasonal_NEEobservations, seasonal_df_subset_inversion), axis=1)
seasonal_df = seasonal_df.apply(scale_fun, axis=0)

# Generate a dictionary mapping column names to corresponding colors
color_dict = {}
for column in seasonal_df.columns:
    if column in high_model_subset:
        color_dict[column] = '#396BB8'  # High skill TRENDY TBMs
    elif column in low_model_subset:
        color_dict[column] = '#D4631D'  # Low skill TRENDY TBMs
    elif column in ['X-BASE', 'ABCflux']:
        color_dict[column] = '#56983F'  # Upscaled fluxes
    elif column in inversion_names:
        color_dict[column] = 'black'  # Inversions

# Separate data into two groups for plotting
group1 = ['ABCflux'] + high_model_subset + inversion_names
group2 = ['X-BASE'] + low_model_subset

fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharey=True)

# Plot for group 1
for column in group1:
    if column in seasonal_df.columns:
        axes[0].plot(np.arange(4,12), seasonal_df[column][3:11], label=column, color=color_dict[column], alpha=0.8)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0].set_xlim(3.5,11.5)
axes[0].set_ylim(ylim)
axes[0].set_xticks(np.arange(4, 12))
axes[0].set_yticks(yticks)
axes[0].tick_params(axis='both', which='major', labelsize=20, direction='in')

# Plot for group 2
for column in group2:
    if column in seasonal_df.columns:
        axes[1].plot(np.arange(4,12), seasonal_df[column][3:11], label=column, color=color_dict[column], alpha=0.8)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].set_xlim(3.5,11.5)
axes[1].set_ylim(ylim)
axes[1].set_xticks(np.arange(4, 12))
axes[1].set_yticks(yticks)
axes[1].tick_params(axis='both', which='major', labelsize=20, direction='in')

# Set shared labels
fig.text(0.6, -0.03, 'Month', ha='center', fontsize=22)
fig.text(-0.03, 0.5, f'Rescaled {ylabel}', va='center', rotation='vertical', fontsize=22)

plt.tight_layout()
plt.show()