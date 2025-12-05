'''calculate the background values for arctic-cap airborne measurements, based on Miller et al. (2016)'''

import pandas as pd
import numpy as np

data = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/arctic_cap_airborne/atm_obs/ABoVE_2017_arctic_cap_airborne_matching_footprint.csv')
data_2000m = data[data['footprint_agl']>2000]
data_2000m.to_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/arctic_cap_airborne/atm_obs/ABoVE_2017_arctic_cap_airborne_matching_footprint_agl2000m.csv', encoding='utf-8', index=False)


# extract unique dates with measurements >2000m agl 
datelist = []
unique_datelist = []
for index, row in data_2000m.iterrows():
    date = pd.Timestamp(row['footprint_time_AKT']).date()
    datelist.append(date)
    if date not in unique_datelist:
        unique_datelist.append(date)

# for each date, take the average of all the CO2 concentration measurements during that day
background_df = pd.DataFrame([])
for date in unique_datelist:
    print(date)
    indices = [i for i, x in enumerate(datelist) if x == date]
    tmp_CO2 = [data_2000m.iloc[[i]]['airborne_CO2'] for i in indices]
    CO2 = np.mean(tmp_CO2)
    CO2_std = np.std(tmp_CO2)
    tmp_CO = [data_2000m.iloc[[i]]['airborne_CO'] for i in indices]
    CO = np.mean(tmp_CO)
    CO_std = np.std(tmp_CO)
    n = len(tmp_CO2)

    single_date = pd.DataFrame({'date': date,
                                'CO2': CO2,
                                'CO2_std': CO2_std,
                                'CO': CO,
                                'CO_std': CO_std,
                                'n': n}, index=[0])
    background_df = pd.concat([background_df, single_date])

background_df.to_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/arctic_cap_airborne/atm_obs/ABoVE_2017_arctic_cap_airborne_background.csv', encoding='utf-8', index=False)


#################################################################
# plot
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

background_df = pd.read_csv('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/arctic_cap_airborne/atm_obs/ABoVE_2017_arctic_cap_airborne_background.csv')

background_df['date'] = pd.to_datetime(background_df['date'])
plt.plot(background_df['date'], background_df['CO2'], c='red')
plt.errorbar(background_df['date'], background_df['CO2'], yerr=background_df['CO2_std'], fmt="o", color="r")
plt.ylabel('CO2 background (ppm)')
# plt.xticks(rotation='vertical')
plt.show()

plt.plot(background_df['date'], background_df['CO'], c='black')
plt.errorbar(background_df['date'], background_df['CO'], yerr=background_df['CO_std'], fmt="o", color="r")
plt.ylabel('CO background (ppb)')
plt.show()