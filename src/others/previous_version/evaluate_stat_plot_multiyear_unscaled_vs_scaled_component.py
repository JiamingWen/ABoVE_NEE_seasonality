# plot correlation of transported GPP, NPP, Reco against airborne measurements

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stat_var = 'cor'; xlim = [0, 1]

# unscaled variables (without linear regression)
fitting_df = pd.DataFrame()
fitting_df_TRENDYv11NEE_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_TRENDYv11.csv')

fitting_df_TRENDYv11GPP_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_TRENDYv11GPP.csv')
fitting_df_TRENDYv11GPP_unscaled[stat_var] = abs(fitting_df_TRENDYv11GPP_unscaled[stat_var])

fitting_df_TRENDYv11NPP_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_TRENDYv11NPP.csv')
fitting_df_TRENDYv11NPP_unscaled[stat_var] = abs(fitting_df_TRENDYv11NPP_unscaled[stat_var])

fitting_df_TRENDYv11Reco_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_TRENDYv11Reco.csv')

fitting_df_TRENDY_unscaled = pd.concat([fitting_df_TRENDYv11NEE_unscaled['model_name'], 
                                fitting_df_TRENDYv11NEE_unscaled[stat_var].rename(f'{stat_var}_NEE'), 
                                fitting_df_TRENDYv11GPP_unscaled[stat_var].rename(f'{stat_var}_GPP'), 
                                fitting_df_TRENDYv11NPP_unscaled[stat_var].rename(f'{stat_var}_NPP'), 
                                fitting_df_TRENDYv11Reco_unscaled[stat_var].rename(f'{stat_var}_Reco')], axis=1)


# other datasets
fitting_df_NEEobservations_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_NEEobservations.csv')

fitting_df_GPPobservations_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_GPPobservations.csv')
fitting_df_GPPobservations_unscaled[stat_var] = abs(fitting_df_GPPobservations_unscaled[stat_var])

fitting_df_reference_unscaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_unscaled_reference.csv')
fitting_df_reference_unscaled[stat_var] = abs(fitting_df_reference_unscaled[stat_var])

fitting_df_unscaled = pd.concat([fitting_df_NEEobservations_unscaled[['model_name', stat_var]], 
                                 fitting_df_GPPobservations_unscaled[['model_name', stat_var]], 
                                 fitting_df_reference_unscaled[['model_name', stat_var]]], axis=0)


# scaled variables (with linear regression)
fitting_df_TRENDYv11NEE_scaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_TRENDYv11.csv')
fitting_df_TRENDYv11GPP_scaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_TRENDYv11GPP.csv')
fitting_df_TRENDYv11NPP_scaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_TRENDYv11NPP.csv')
fitting_df_TRENDYv11Reco_scaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_TRENDYv11Reco.csv')

fitting_df_TRENDY_scaled = pd.concat([fitting_df_TRENDYv11NEE_scaled['model_name'], 
                                fitting_df_TRENDYv11NEE_scaled[stat_var].rename(f'{stat_var}_NEE'), 
                                fitting_df_TRENDYv11GPP_scaled[stat_var].rename(f'{stat_var}_GPP'), 
                                fitting_df_TRENDYv11NPP_scaled[stat_var].rename(f'{stat_var}_NPP'), 
                                fitting_df_TRENDYv11Reco_scaled[stat_var].rename(f'{stat_var}_Reco')], axis=1)

# other datasets
fitting_df_NEEobservations_scaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_NEEobservations.csv')
fitting_df_GPPobservations_scaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_GPPobservations.csv')

fitting_df_reference_scaled = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/regression/evaluation_stat_reference.csv')

fitting_df_scaled = pd.concat([fitting_df_NEEobservations_scaled[['model_name', stat_var]], 
                                 fitting_df_GPPobservations_scaled[['model_name', stat_var]], 
                                 fitting_df_reference_scaled[['model_name', stat_var]]], axis=0)



# merge two dataframes
fitting_df_TRENDY = pd.merge(fitting_df_TRENDY_unscaled, fitting_df_TRENDY_scaled, on='model_name', how='outer', suffixes=('_unscaled', '_scaled'))
fitting_df = pd.merge(fitting_df_unscaled, fitting_df_scaled, on='model_name', how='outer', suffixes=('_unscaled', '_scaled'))


# sort for each category
fitting_df_TRENDY_sorted = fitting_df_TRENDY.sort_values(f'{stat_var}_NEE_unscaled')

fitting_df_NEEobservations_sorted = fitting_df[fitting_df['model_name'] .isin (fitting_df_NEEobservations_scaled['model_name'])].sort_values(f'{stat_var}_unscaled')
fitting_df_GPPobservations_sorted = fitting_df[fitting_df['model_name'] .isin (fitting_df_GPPobservations_scaled['model_name'])].sort_values(f'{stat_var}_unscaled')
fitting_df_reference_sorted = fitting_df[fitting_df['model_name'] .isin (fitting_df_reference_scaled['model_name'])].sort_values(f'{stat_var}_unscaled')

fitting_df_sorted = pd.concat((fitting_df_NEEobservations_sorted, fitting_df_GPPobservations_sorted, fitting_df_reference_sorted), axis=0)


# plot
fig = plt.figure(figsize=(5,8))
plt.scatter(abs(fitting_df_TRENDY_sorted[f'{stat_var}_NEE_unscaled']), fitting_df_TRENDY_sorted['model_name'], marker='o', color='black', facecolors='none', label='NEE w/o reg')
plt.scatter(abs(fitting_df_TRENDY_sorted[f'{stat_var}_GPP_unscaled']), fitting_df_TRENDY_sorted['model_name'], marker='o', color='b', facecolors='none', label='GPP w/o reg')
plt.scatter(abs(fitting_df_TRENDY_sorted[f'{stat_var}_NPP_unscaled']), fitting_df_TRENDY_sorted['model_name'], marker='o', color='r', facecolors='none', label='NPP w/o reg')
plt.scatter(abs(fitting_df_TRENDY_sorted[f'{stat_var}_Reco_unscaled']), fitting_df_TRENDY_sorted['model_name'], marker='o', color='g', facecolors='none', label='Reco w/o reg')
plt.scatter(abs(fitting_df_sorted[f'{stat_var}_unscaled']), fitting_df_sorted['model_name'], marker='s', color='black', facecolors='none')

# plt.scatter(abs(fitting_df_TRENDY_sorted[f'{stat_var}_NEE_scaled']), fitting_df_TRENDY_sorted['model_name'], marker='o', color='black', label='NEE w/ reg')
# plt.scatter(abs(fitting_df_TRENDY_sorted[f'{stat_var}_GPP_scaled']), fitting_df_TRENDY_sorted['model_name'], marker='o', color='b', label='GPP w/ reg')
# plt.scatter(abs(fitting_df_TRENDY_sorted[f'{stat_var}_NPP_scaled']), fitting_df_TRENDY_sorted['model_name'], marker='o', color='r', label='NPP w/ reg')
# plt.scatter(abs(fitting_df_TRENDY_sorted[f'{stat_var}_Reco_scaled']), fitting_df_TRENDY_sorted['model_name'], marker='o', color='g', label='Reco w/ reg')
# plt.scatter(abs(fitting_df_sorted[f'{stat_var}_scaled']), fitting_df_sorted['model_name'], marker='s', color='black')

plt.axhline(y = 15.5, color = 'grey', linestyle = '--')
plt.axhline(y = 18.5, color = 'grey', linestyle = '--')

plt.xlim(xlim)
plt.ylim(-1, fitting_df_TRENDY_sorted.shape[0]+fitting_df_sorted.shape[0])
plt.xlabel(f'{stat_var} in absolute values', fontsize=15)
plt.xticks(fontsize=15) #np.arange(-0.2, 1, 0.2), 
plt.yticks(fontsize=15)
plt.legend(loc='lower right')


plt.savefig(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/result/evaluation_stat_multiyear_component_{stat_var}.png', dpi=100, bbox_inches='tight')
plt.show()