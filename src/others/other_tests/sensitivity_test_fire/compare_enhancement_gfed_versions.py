'''compare CO2 enhancement of fire emissions from different versions'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
import sys
sys.path.append('/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src')
from functions import get_campaign_info

for year in [2012, 2013, 2014, 2017]: #2012, 2013, 2014, 2017

    start_month, end_month, campaign_name = get_campaign_info(year)
    month_num = end_month - start_month + 1

    df_fire_year = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/transported_surface_field/ABoVE_{year}_{campaign_name}_airborne_fire.csv')

    if year == 2012:
        df_fire = df_fire_year
    else:
        df_fire = pd.concat((df_fire, df_fire_year))

y = df_fire['gfed5']
x = df_fire['gfed4.1']
pearson_res = pearsonr(y, x)
cor, _ = pearson_res
mean_bias = np.mean(x - y)
rmse = np.sqrt(np.mean((x - y)**2))

y_tmp = sm.add_constant(y) # observation as x-axis, modeled as y-axis
model = sm.OLS(x, y_tmp)
results1 = model.fit()
slope = results1.params.iloc[1]
intercept = results1.params.iloc[0]

fig, ax = plt.subplots(figsize=(4, 4))
xy = np.vstack([y, x])
z = stats.gaussian_kde(xy)(xy)
idx = z.argsort()
x_sorted, y_sorted, z_sorted = y.iloc[idx], x.iloc[idx], z[idx]
plt.scatter(x_sorted, y_sorted, c=z_sorted, s=60, edgecolors="none", cmap='viridis')

x_line = np.arange(-50,50,0.1)
plt.plot(x_line, x_line, color='black', linestyle='dashed') # Plot 1:1 line
y_line = x_line * slope + intercept
plt.plot(x_line, y_line, color='red', linestyle='dashed') # regression line
plt.annotate(r"$Cor$ = {:.2f}".format(cor), (45, -32), fontsize=14, ha='right', va='bottom')
plt.annotate('Bias = {:.2f}'.format(mean_bias), (45, -40), fontsize=14, ha='right', va='bottom')
plt.annotate('y={:.2f}x{:+.2f}'.format(slope, intercept), (45, -48), fontsize=14, ha='right', va='bottom')

plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel('CO$_2$ enhancement from v4.1 (ppm)', fontsize=13.5)
ax.set_ylabel('CO$_2$ enhancement from v5 (ppm)', fontsize=13.5)
plt.savefig(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/other/sensitivity_test_fire/gfed_versions_scatterplot.png", dpi=100, bbox_inches='tight')
plt.show()