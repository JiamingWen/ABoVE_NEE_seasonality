'''plot an example of NEE seasonal cycle to illustrate the seasonal features'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

option = 'cor' #none mean_bias range_ratio cor

ylim1 = -4
ylim2 = 2

model_type = 'TRENDYv11'
model_name = 'ISBA-CTRIP' # ISBA-CTRIP SDGVM LPX-Bern

weightname = 'unweighted' #unweighted weighted
lcname = 'alllc' #alllc forestshrub forest shrub tundra
regionname = 'ABoVEcore'

seasonal_df = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_{model_type}_{regionname}_{lcname}_{weightname}.csv')
fig, ax = plt.subplots(figsize=(3,3))
plt.plot(np.arange(4,12), seasonal_df[model_name][3:11], linestyle='-', linewidth=5, label='Model 1') #'#d4631d'

if option == 'mean_bias':
    plt.plot(np.arange(4,12), seasonal_df[model_name][3:11]-1, linestyle=':', linewidth=5, color='#d4631d', label='Model 2') #'#d4631d'
elif option == 'range_ratio':
    plt.plot(np.arange(4,12), seasonal_df[model_name][3:11]*2, linestyle=':', linewidth=5, color='#d4631d', label='Model 2') #'#d4631d'
elif option == 'cor':
    plt.plot(np.arange(4,12), seasonal_df[model_name][4:12], linestyle=':', linewidth=5, color='#d4631d', label='Model 2') #'#d4631d'
    
if option != 'none':
    plt.legend(loc='lower right')

plt.xlabel('Month', fontsize=18)
plt.ylabel(f'NEE ' + '($\mu$mol m$^{-2}$ s$^{-1}$)', fontsize=18)
ax.set_xlim(3.5,11.5)
ax.set_xticks(np.arange(4,12))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(ylim1, ylim2)
plt.show()

# print(np.mean(seasonal_df[model_name]))
# print(np.mean(seasonal_df[model_name][3:11]))
