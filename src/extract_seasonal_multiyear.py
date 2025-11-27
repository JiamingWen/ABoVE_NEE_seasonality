'''
calculate multiyear average seasonal cycle of different models
based on results from extract_seasonal.py
'''

import pandas as pd
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info


dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal"

for regionname in ['ABoVEcore']: #, 'ABoVEcoreextended'

    for lcname in ['alllc']: #'alllc', 'forest', 'shrub', 'tundra'

        for weightname in ['unweighted']: #, 'weighted'

            for model_type in ['TRENDYv11', 'inversions', 'UpscaledEC', 'reference', 'TRENDYv11GPP', 'TRENDYv11Ra', 'TRENDYv11Rh', 'TRENDYv11LAI', 'UpscaledEC_GPP', 'GPPobservations', 'UpscaledEC_Reco', 'fossil', 'fire', 'inversionsNEE', 'inversions-prior', 'inversionsNEE-prior']:
                #'TRENDYv11', 'inversions', 'UpscaledEC', 'reference', 'TRENDYv11GPP', 'TRENDYv11Ra', 'TRENDYv11Rh', 'TRENDYv11LAI', 'UpscaledEC_GPP', 'GPPobservations', 'UpscaledEC_Reco', 'fossil', 'fire', 'inversionsNEE', 'inversions-prior', 'inversionsNEE-prior'
                
                print(regionname, lcname, weightname, model_type)

                for year in [2012, 2013, 2014, 2017]:

                    start_month, end_month, campaign_name = get_campaign_info(year)
                    seasonal_df_year = pd.read_csv(f'{dir0}/seasonal_{year}_{model_type}_{regionname}_{lcname}_{weightname}.csv')

                    if year == 2012:
                        seasonal_df = seasonal_df_year
                    else:
                        seasonal_df = seasonal_df + seasonal_df_year

                seasonal_df = seasonal_df/4 # calculate the average for 4 years
                seasonal_df.to_csv(f'{dir0}/seasonal_{model_type}_{regionname}_{lcname}_{weightname}.csv', encoding='utf-8', index=False)
