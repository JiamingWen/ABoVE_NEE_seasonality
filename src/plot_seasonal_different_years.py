'''
check how seasonal cycle of TRENDY and inversion models varies 
across years during 2012, 2013, 2014, 2017
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info


weightname = 'unweighted' #unweighted weighted
lcname = 'alllc'
regionname = 'ABoVEcore'

ylim1 = -2.5
ylim2 = 2.5

for model_type in ['TRENDYv11', 'inversionsNEE']: #, 'TRENDYv9', 'inversions'
    if model_type == 'TRENDYv11':
        model_names = ['CABLE-POP', 'CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'JULES', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT', 'VISIT-NIES', 'YIBs']
        fig = plt.figure(figsize=(22,15))
        ncols = 4
    elif model_type == 'TRENDYv9':
        model_names = ['CLASSIC', 'CLM5.0', 'IBIS', 'ISAM', 'ISBA-CTRIP', 'JSBACH', 'LPJ', 'LPX-Bern', 'OCN', 'ORCHIDEE', 'SDGVM', 'VISIT']
        fig = plt.figure(figsize=(22,11))
        ncols = 4
    elif model_type in ['inversions', 'inversionsNEE']:
        # model_names = ['CAMS', 'CAMS-Satellite', 'CarboScope', 'CMS-Flux', 'COLA', 'CTE', 'CT-NOAA', 'GCASv2', 'GONGGA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'THU', 'UoE']
        # fig = plt.figure(figsize=(22,15))
        model_names = ['CAMS', 'CarboScope', 'CMS-Flux', 'CTE', 'CT-NOAA', 'IAPCAS', 'MIROC', 'NISMON-CO2', 'UoE'] # excluding models without CARVE coverage
        fig = plt.figure(figsize=(15,10))
        ncols = 3

    subplot_id = 0
    for model_name in model_names:
        subplot_id += 1
        ax1 = plt.subplot(np.ceil(len(model_names)/ncols).astype(int),ncols,subplot_id) #,aspect='equal'

        for year, color in zip([2012, 2013, 2014, 2017], ['blue', 'orange', 'green', 'purple']):

            campaign_name = get_campaign_info(year)[2]
            dir0 = "/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal"
            
            seasonal_df = pd.read_csv(f'{dir0}/seasonal_{year}_{model_type}_{regionname}_{lcname}_{weightname}.csv')
            seasonal_df = seasonal_df.replace(0, -9999)
            
            # plot Apr - Nov
            plt.plot(np.arange(4,12), seasonal_df[model_name][3:11], linestyle='-',color=color, label=year)
            
            plt.ylabel(f'NEE ' + '($\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=15)
            ax1.set_xticks(np.arange(4,12))
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            # plt.title(model_name, fontsize=20)
            plt.ylim(ylim1, ylim2)

            if model_name == 'MIROC':
                title_name = 'MIROC4-ACTM'
            else:
                title_name = model_name

            subtitle = chr(ord('`')+subplot_id+1)
            ax1.annotate(f'({subtitle}) {title_name}', xy=(0.95, 0.05), xycoords='axes fraction', fontsize=18, ha='right', va='bottom')

            if model_name == 'LPX-Bern':
                plt.ylim(-5,5)
            
        if subplot_id == 1:
            plt.legend(ncol=2, fontsize=15)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f'{dir0}/plot/seasonal_different_years_{model_type}_{regionname}_{lcname}_{weightname}.png', dpi=100, bbox_inches='tight')
    
    # if model_type == 'inversionsNEE':
    #     plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/seasonal_different_years_{model_type}_{regionname}_{lcname}_{weightname}.png', dpi=300, bbox_inches='tight')
    #     plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/seasonal_different_years_{model_type}_{regionname}_{lcname}_{weightname}.pdf', dpi=300, bbox_inches='tight')
        
    plt.show()

