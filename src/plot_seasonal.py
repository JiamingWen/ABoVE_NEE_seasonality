import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info


weightname = 'unweighted' #unweighted weighted
lcname = 'alllc' #alllc forestshrub forest shrub tundra
regionname = 'ABoVEcore'

ylim1 = -2.5
ylim2 = 2.5

for year in [2012, 2013, 2014, 2017]: #

    start_month, end_month, campaign_name = get_campaign_info(year)
    dir0 = '/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal'

    for model_type in ['TRENDYv11', 'inversionsNEE']: #, 'TRENDYv9', 'inversions
        if model_type == 'TRENDYv11':
            model_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
            fig = plt.figure(figsize=(22,15))
        elif model_type == 'TRENDYv9':
            model_names = ['CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT']
            fig = plt.figure(figsize=(22,11))
        elif model_type == 'inversionsNEE':
            model_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
            fig = plt.figure(figsize=(22,15))
        
        seasonal_df = pd.read_csv(f'{dir0}/seasonal_{year}_{model_type}_{regionname}_{lcname}_{weightname}.csv')
        seasonal_df = seasonal_df.replace(0, -9999)

        subplot_id = 0

        for model_name in model_names:
            subplot_id += 1
            plt.subplot(np.ceil(len(model_names)/4).astype(int),4,subplot_id,aspect='equal')

            # plot Apr - Nov
            plt.plot(np.arange(4,12), seasonal_df[model_name][3:11], linestyle='-',color='black')
            plt.ylabel(f'NEE ' + '($\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.title(model_name, fontsize=20)
            plt.ylim(ylim1, ylim2)

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        # plt.tight_layout()
        plt.savefig(f'{dir0}/plot/seasonal_{year}_{model_type}_{regionname}_{lcname}_{weightname}.png', dpi=100, bbox_inches='tight')
        plt.show()

