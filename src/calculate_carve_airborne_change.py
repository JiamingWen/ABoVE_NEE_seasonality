# calculate carve airborne enhancement and drawdown

import pandas as pd
import numpy as np

year = 2014 # 2012 2013 2014

data = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/carve_airborne/atm_obs/ABoVE_{year}_carve_airborne_matching_footprint.csv')
data_near_surface = data[data['footprint_agl'] <= 2000]
data_background = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/carve_airborne/atm_obs/ABoVE_{year}_carve_airborne_background.csv')


# for each observation, subtract the background value of that day if available
list_of_dataframes = []
for index, row in data_near_surface.iterrows():

    date = str(pd.Timestamp(row['footprint_time_AKT']).date())
    id = data_background.index[data_background['date'] == date].tolist()
    print(id)

    if len(id) == 1:
        row_background = data_background.loc[data_background.index[id]]
        row['background_CO2'] = float(row_background['CO2'].iloc[0])
        row['background_CO2_std'] = float(row_background['CO2_std'].iloc[0])
        row['background_CO2_n'] = float(row_background['n'].iloc[0])
        row['CO2_change'] = row['airborne_CO2'] - row['background_CO2']

        row['background_CO'] = float(row_background['CO'].iloc[0])
        row['background_CO_std'] = float(row_background['CO_std'].iloc[0])
        row['CO_change'] = row['airborne_CO'] - row['background_CO']
    else:
        print('background value not found')
        
    list_of_dataframes.append(row)

CO2_change_df = pd.DataFrame(list_of_dataframes)
CO2_change_df.to_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/carve_airborne/atm_obs/ABoVE_{year}_carve_airborne_change.csv', encoding='utf-8', index=False)



# # check the calculation
# CO2_change_df['CO2_change'].describe()
# (CO2_change_df['CO2_change']/CO2_change_df['background_CO2_std']).describe()

# #plot
# from matplotlib import pyplot as plt
# from datetime import datetime
# import pandas as pd
# CO2_change_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/carve_airborne/atm_obs/ABoVE_{year}_carve_airborne_change.csv')
# CO2_change_df['hour'] = [datetime.strptime(CO2_change_df['footprint_time_AKT'][index], "%Y-%m-%d %H:%M:%S-0%f:00").hour for index, row in CO2_change_df.iterrows()]
# CO2_change_df['minute'] = [datetime.strptime(CO2_change_df['footprint_time_AKT'][index], "%Y-%m-%d %H:%M:%S-0%f:00").minute for index, row in CO2_change_df.iterrows()]
# CO2_change_df['month'] = [datetime.strptime(CO2_change_df['footprint_time_AKT'][index], "%Y-%m-%d %H:%M:%S-0%f:00").month for index, row in CO2_change_df.iterrows()]
# CO2_change_df['time_of_day'] = CO2_change_df['hour'].astype(float) + CO2_change_df['minute'].astype(float)/60

# fig, ax = plt.subplots()
# scatter = ax.scatter(CO2_change_df['time_of_day'], CO2_change_df['CO2_change'], c=CO2_change_df['month'],s=5, cmap="Spectral")
# plt.axis()
# plt.ylabel('CO2 change (ppm)')
# plt.xlabel('Time of day')
# legend1 = ax.legend(*scatter.legend_elements(),
#                      title="Month")
# ax.add_artist(legend1)
# plt.show()


# # plot vertical profiles near Fairbanks
# df = CO2_change_df
# df['month'] = df['month'].astype('category')
# df_sub = df[(df['airborne_lat'] > 64.5) & (df['airborne_lat'] < 65) & (df['airborne_lon'] < -147.5) & (df['airborne_lon'] > -148)]
# fig, ax = plt.subplots()
# scatter = ax.scatter(df_sub.CO2_change, df_sub.airborne_alt, c=df_sub.month, cmap="Spectral")
# # labels = df_sub.month.cat.categories
# plt.axis()
# plt.ylabel('Altitude (meters)')
# plt.xlabel('CO2 change (ppm)')
# # plt.legend(handles=scatter.legend_elements()[0], labels=labels)
# legend1 = ax.legend(*scatter.legend_elements(num=8),
#                     loc="upper right", title="Month")
# ax.add_artist(legend1)
# plt.show()


# # plot spatial map
# CO2_change_df['month'] = CO2_change_df['month'].astype('category')
# fig, ax = plt.subplots()
# # scatter = ax.scatter(CO2_change_df['airborne_lon'], CO2_change_df['airborne_lat'], c=CO2_change_df['CO2_change'],s=5, cmap="Spectral")
# scatter = ax.scatter(CO2_change_df['airborne_lon'], CO2_change_df['airborne_lat'], c=CO2_change_df['month'],s=5, cmap="Spectral")
# plt.axis()
# plt.ylabel('lat')
# plt.xlabel('lon')
# # legend1 = ax.legend(*scatter.legend_elements(),
# #                      title="CO2 change")
# legend1 = ax.legend(*scatter.legend_elements(),
#                      title="Month")
# ax.add_artist(legend1)
# plt.show()


# # plot coordinates by months
# fig = plt.figure(figsize=(18,9))
# subplot_id = 0
# for month_num in np.arange(4,12):
#     subplot_id += 1
#     plt.subplot(2,4,subplot_id)

#     month_list = [month_num] #4, 5, 7, 8, 9
#     CO2_change_df_subset = CO2_change_df[CO2_change_df.month.isin(month_list)]
#     scatter = plt.scatter(CO2_change_df_subset['airborne_lon'], CO2_change_df_subset['airborne_lat'],s=5)
#     plt.axis()
#     plt.ylabel('lat')
#     plt.xlabel('lon')
#     plt.title('Month: '+str(month_num))
# plt.show()