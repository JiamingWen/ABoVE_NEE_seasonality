'''compare annual or growing season NEE of prior and posterior estimates for inversions in Alaska'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

regionname = 'AK' # 'AK', 'forest_AK', 'tundra_NS', 'tundra_YK', 'others_AK'

for year in [2012, 2013, 2014, 2017]:

    # seasonal cycle
    if year in [2012, 2013, 2014]: # all years - only five models
        model_names = ['CAMS', 'CMS-Flux', 'CTE', 'MIROC', 'NISMON-CO2']
    elif year == 2017: # 2017
        model_names = ['CAMS', 'CAMS-Satellite', 'CMS-Flux', 'CTE', 'GCASv2', 'GONGGA', 'MIROC', 'NISMON-CO2', 'THU']

    for (model_id, model_name) in enumerate(model_names):
        seasonal_df_prior = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_magnitude/seasonal/seasonal_{year}_inversionsNEE-prior_{regionname}.csv')
        seasonal_df_posterior = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_magnitude/seasonal//seasonal_{year}_inversionsNEE_{regionname}.csv')

        # May - Oct
        prior = np.mean(seasonal_df_prior.loc[[4,5,6,7,8,9], model_name])
        posterior = np.mean(seasonal_df_posterior.loc[[4,5,6,7,8,9], model_name]) # growing season: May to October

        # # Apr - Nov
        # prior = np.mean(seasonal_df_prior.loc[[3,4,5,6,7,8,9,10], model_name])
        # posterior = np.mean(seasonal_df_posterior.loc[[3,4,5,6,7,8,9,10], model_name]) # growing season: April to November


        annuan_NEE_df0 = pd.DataFrame([[model_name, prior, posterior]], 
                                columns=['model_name', 'prior', 'posterior'])
        if model_id == 0:
            annuan_NEE_df = annuan_NEE_df0
        else:
            annuan_NEE_df = pd.concat((annuan_NEE_df, annuan_NEE_df0))

    annuan_NEE_df = annuan_NEE_df.sort_values('model_name', ascending=False)

    if year in [2012, 2013, 2014]:
        fig, ax = plt.subplots(figsize=(7,3))
    elif year == 2017:
        fig, ax = plt.subplots(figsize=(7,4))

    plt.scatter(annuan_NEE_df['posterior'], annuan_NEE_df['model_name'], marker='s', color='black', facecolor='none', s=60, label='posterior')
    plt.scatter(annuan_NEE_df['prior'], annuan_NEE_df['model_name'], marker='s', color='red', facecolor='none', s=60, label='prior')

    plt.axvline(x=0, color='purple', linestyle='--')
    plt.legend(fontsize=15)

    xlim = [-0.8, 0.2]
    plt.xlim(xlim)
    plt.ylim(-1, annuan_NEE_df.shape[0]-0.5)
    plt.xlabel(f'Growing season NEE ' + '($\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=18)
    plt.xticks(ticks=np.arange(xlim[0], xlim[1], 0.1), fontsize=15) #np.arange(-0.2, 1, 0.2), 
    plt.yticks(fontsize=15)
    plt.title(f'{year}', fontsize=20)

    plt.show()