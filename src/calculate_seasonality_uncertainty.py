'''calculate and plot seasonality difference between each pair of TRENDY models'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

weightname = 'unweighted' #unweighted
regionname = 'ABoVEcore'

high_model_subset = ['ISBA-CTRIP', 'LPJ', 'CLASSIC', 'CLM5.0']
low_model_subset = ['ORCHIDEE', 'JULES', 'OCN', 'VISIT', 'JSBACH', 'LPX-Bern', 'SDGVM', 'VISIT-NIES', 'YIBs', 'CABLE-POP', 'ISAM']
all_model = high_model_subset + low_model_subset

# standardize with minumum NEE
def scale_minumum (vec):
    return -vec / np.min(vec)

# standardize with maximum and minimum
def scale_maximum_minimum (vec):
    return (vec - np.min(vec)) / (np.max(vec) - np.min(vec))

# MAD
def cal_MAD (vec1, vec2):
    return np.mean(np.abs(vec1 - vec2))

for varname in ['NEE', 'GPP', 'Reco', 'Ra', 'Rh']:
    
    if varname == 'NEE':
        scale_fun = scale_minumum
        filestr = ''

    else:
        scale_fun = scale_maximum_minimum
        filestr = varname

    if varname == 'Reco':
        seasonal_df_TRENDYv11Ra = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Ra_{regionname}_{lcname}_{weightname}.csv")
        seasonal_df_TRENDYv11Rh = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11Rh_{regionname}_{lcname}_{weightname}.csv")
        seasonal_df_TRENDYv11 = seasonal_df_TRENDYv11Ra + seasonal_df_TRENDYv11Rh
    else:
        seasonal_df_TRENDYv11 = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11{filestr}_{regionname}_{lcname}_{weightname}.csv")
    
    seasonal_df_TRENDYv11 = seasonal_df_TRENDYv11.apply(scale_fun, axis=0)

    # calculate pairwise MAD during the growing season
    MAD_df = pd.DataFrame(columns=['model1', 'model2', 'MAD'])

    for model1 in all_model:
        for model2 in all_model:
            if model1 == model2:
                continue
            MAD_pairwise = cal_MAD(seasonal_df_TRENDYv11[model1][3:11], seasonal_df_TRENDYv11[model2][3:11])
            MAD_df = pd.concat([MAD_df, pd.DataFrame([{'model1': model1, 'model2': model2, 'MAD': MAD_pairwise}])], ignore_index=True)
    
    MAD_df.to_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonality_uncertainty/{varname}_MADpairwise.csv", index=False)



'''plot'''
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

colors = ['black','#50c878', '#e573d5','#d4bf1d', '#1367e9']

def customize_boxplot(bplot, colors):
    for i, box in enumerate(bplot['boxes']):
        box.set(linewidth=2, edgecolor=colors[i], facecolor='none')
    for i, whisker in enumerate(bplot['whiskers']):
        whisker.set(linewidth=2, color=colors[i // 2])
    for i, cap in enumerate(bplot['caps']):
        cap.set(linewidth=2, color=colors[i // 2])
    for i, median in enumerate(bplot['medians']):
        median.set(linewidth=2, color=colors[i])

# Panel (a) MAD of NEE, GPP, Reco, Ra, Rh for all models
all_mad_data = []
for varname in ['NEE', 'GPP', 'Reco', 'Ra', 'Rh']:
    mad_df = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonality_uncertainty/{varname}_MADpairwise.csv")
    all_mad_data.append(mad_df['MAD'])

bplot1 = axes[0].boxplot(all_mad_data, labels=['NEE', 'GPP', 'Reco', 'Ra', 'Rh'], patch_artist=True, showfliers=False)
customize_boxplot(bplot1, colors)

axes[0].set_ylabel('Pairwise MAD', fontsize=16)
axes[0].set_ylim(0, 0.65)
axes[0].tick_params(axis='both', which='major', labelsize=15)
axes[0].text(0.95, 0.95, '(a) All TBMs', transform=axes[0].transAxes, fontsize=16, verticalalignment='top', horizontalalignment='right')

# Panel (b) MAD of NEE, GPP, Reco, Ra, Rh for high_model_subset
high_mad_data = []
for varname in ['NEE', 'GPP', 'Reco', 'Ra', 'Rh']:
    mad_df = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonality_uncertainty/{varname}_MADpairwise.csv")
    high_mad_df = mad_df[mad_df['model1'].isin(high_model_subset) & mad_df['model2'].isin(high_model_subset)]
    high_mad_data.append(high_mad_df['MAD'])

bplot2 = axes[1].boxplot(high_mad_data, labels=['NEE', 'GPP', 'Reco', 'Ra', 'Rh'], patch_artist=True, showfliers=False)
customize_boxplot(bplot2, colors)

axes[1].set_ylabel('Pairwise MAD', fontsize=16)
axes[1].set_ylim(0, 0.65)
axes[1].tick_params(axis='both', which='major', labelsize=15)
axes[1].text(0.95, 0.95, '(b) High-cor TBMs', transform=axes[1].transAxes, fontsize=16, verticalalignment='top', horizontalalignment='right')

plt.tight_layout()

plt.savefig("/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/seasonal_uncertainty.png", dpi=300, bbox_inches='tight')
plt.savefig("/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/figures/seasonal_uncertainty.png", dpi=300, bbox_inches='tight')
plt.show()
