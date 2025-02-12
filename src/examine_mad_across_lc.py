'''plot relationship of NEE seasonality bias between the whole region and individual land covers'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

weightname = 'unweighted' #unweighted
regionname = 'ABoVEcore'
reference = 'GroupH' # GroupH obs
varname = 'NEE' # NEE GPP Ra Rh Reco

high_model_subset = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_model_subset = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM']

# mad data (calculated from Fig3.py)
mad_alllc = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonality_diff/seasonal_mad_{reference}_{varname}_{regionname}_{weightname}.csv')
mad_forest = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonality_diff/seasonal_mad_{reference}_{varname}_{regionname}_forest_{weightname}.csv')
mad_shrub = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonality_diff/seasonal_mad_{reference}_{varname}_{regionname}_shrub_{weightname}.csv')
mad_tundra = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonality_diff/seasonal_mad_{reference}_{varname}_{regionname}_tundra_{weightname}.csv')

# set color
mad_alllc.loc[mad_alllc['model_name'].isin(high_model_subset),'color'] = '#396BB8'
mad_alllc.loc[mad_alllc['model_name'].isin(low_model_subset),'color'] = '#D4631D'
mad_alllc.loc[mad_alllc['model_name'].isin(['IBIS']),'color'] = 'grey'
mad_alllc.loc[mad_alllc['model_name'].isin(['X-BASE', 'ABCflux']),'color'] = '#56983F'
mad_alllc['shape'] = 'o'
mad_alllc.loc[mad_alllc['model_name'].isin(['X-BASE', 'ABCflux']), 'shape'] = 'd'

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
land_covers = ['Forests', 'Shrubs', 'Tundra']
mad_data = [mad_forest, mad_shrub, mad_tundra]
if reference == 'obs':
    subtitles = ['(a)', '(b)', '(c)']
else:
    subtitles = ['(d)', '(e)', '(f)']

for ax, lc, mad, subtitle in zip(axs, land_covers, mad_data, subtitles):
    for i in range(len(mad_alllc)):
        ax.scatter(mad_alllc['mean_seasonal_diff'].iloc[i], mad['mean_seasonal_diff'].iloc[i], color=mad_alllc['color'].iloc[i], marker=mad_alllc['shape'].iloc[i], s=50)
    
    # add a regression line
    df_for_reg = pd.merge(mad_alllc, mad, on='model_name', suffixes=('_alllc', '_lc'))
    x = df_for_reg['mean_seasonal_diff_alllc']
    y = df_for_reg['mean_seasonal_diff_lc']
    m, b = np.polyfit(x, y, 1)
    x0 = np.arange(-0.1,0.9,0.1)
    ax.plot(x0, m * x0 + b, color='k', linestyle='--', alpha=0.8)
    corr, _ = pearsonr(x, y)
    ax.text(0.5, 0.05, f'Cor: {"{:.2f}".format(corr)}', fontsize=15)

    ax.set_xlabel('MAD of all pixels', fontsize=18)
    ax.set_ylabel(f'MAD of {lc.lower()} pixels', fontsize=18)
    ax.set_xlim([0, 0.75])
    ax.set_ylim([0, 0.75])
    ax.set_xticks(np.arange(0.1, 0.76, 0.2))
    ax.set_yticks(np.arange(0.1, 0.76, 0.2))
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.text(0.05, 0.95, f'{subtitle} {lc}', transform=ax.transAxes, fontsize=18, va='top')
plt.tight_layout()

plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/seasonal_mad_lc_{reference}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/seasonal_mad_lc_{reference}.pdf', dpi=300, bbox_inches='tight')
plt.show()
