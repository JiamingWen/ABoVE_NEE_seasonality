'''plot empirical relationships between statistical metrics and seasonal features'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# stat_var = 'cor'; xlim = [-0.2, 0.8]; xlabel = r'Correlation'
stat_var = 'mean_bias'; xlim = [-8, 2]; xlabel = r'Mean bias (ppm)'
# stat_var = 'range_ratio_95_5'; xlim = [0, 3]; xlabel = r'Range ratio'

lcname = 'alllc' #alllc forest shrub tundra
if lcname == 'alllc':
    lc_filestr = ''
elif lcname in ['forest', 'shrub', 'tundra']:
    lc_filestr = '_' + lcname

# # statistics
# fitting_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_background-ct.csv')
# # fitting_df_TRENDYv11 = fitting_df_TRENDYv11.loc[~fitting_df_TRENDYv11['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
# fitting_df_inversions = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_background-ct.csv')
# fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
# fitting_df_UpscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_background-ct.csv')


# fitting_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_background-ct_diurnal_x_base.csv')
# # fitting_df_TRENDYv11 = fitting_df_TRENDYv11.loc[~fitting_df_TRENDYv11['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
# fitting_df_inversions = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_background-ct_diurnal_x_base.csv')
# fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
# fitting_df_UpscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_background-ct_diurnal_x_base.csv')


if stat_var in ['mean_bias', 'range_ratio_95_5']: # imposing X-BASE diurnal cycle
    fitting_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}_background-ct.csv')
    # fitting_df_TRENDYv11 = fitting_df_TRENDYv11.loc[~fitting_df_TRENDYv11['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
    fitting_df_inversions = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}_background-ct.csv')
    fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
    fitting_df_UpscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}_background-ct.csv')
else:
    fitting_df_TRENDYv11 = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_TRENDYv11{lc_filestr}.csv')
    # fitting_df_TRENDYv11 = fitting_df_TRENDYv11.loc[~fitting_df_TRENDYv11['model_name'].isin(['IBIS']), :] # remove IBIS because it simulates negative Rh
    fitting_df_inversions = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_inversionsNEE{lc_filestr}.csv')
    fitting_df_inversions = fitting_df_inversions.loc[~fitting_df_inversions['model_name'].isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU']), :] ## for models with no coverage of CARVE years
    fitting_df_UpscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/evaluation_stat/evaluation_stat_unscaled_UpscaledEC{lc_filestr}.csv')

# seasonal cycle
weightname = 'unweighted' #unweighted weighted
lcname = 'alllc' #alllc forestshrub forest shrub tundra
regionname = 'ABoVEcore'

seasonal_df_TRENDY = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_TRENDYv11_{regionname}_{lcname}_{weightname}.csv')
seasonal_df_inversions = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_inversionsNEE_{regionname}_{lcname}_{weightname}.csv')
seasonal_df_inversions = seasonal_df_inversions.loc[:, ~seasonal_df_inversions.columns.isin(['CAMS-Satellite', 'COLA', 'GCASv2', 'GONGGA', 'THU'])] ## for models with no coverage of CARVE years
seasonal_df_UpscaledEC = pd.read_csv(f'/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/result/seasonal/seasonal_UpscaledEC_{regionname}_{lcname}_{weightname}.csv')

if stat_var == 'mean_bias':
    seasonal_df_metric_TRENDYv11 = seasonal_df_TRENDY.apply(lambda x: np.mean(x[3:11]), axis=0)  # mean seasonal NEE Apr-Nov
    seasonal_df_metric_inversions = seasonal_df_inversions.apply(lambda x: np.mean(x[3:11]), axis=0)
    seasonal_df_metric_UpscaledEC = seasonal_df_UpscaledEC.apply(lambda x: np.mean(x[3:11]), axis=0)
    ylabel = 'NEE seasonal mean\n(µmol m⁻² s⁻¹)'
    reference = 0.0  # reference value for mean bias

elif stat_var == 'range_ratio_95_5':
    seasonal_df_metric_TRENDYv11 = seasonal_df_TRENDY.apply(lambda x: (np.max(x[3:11]) - np.min(x[3:11])), axis=0)  # seasonal amplitude Apr-Nov
    seasonal_df_metric_inversions = seasonal_df_inversions.apply(lambda x: (np.max(x[3:11]) - np.min(x[3:11])), axis=0)
    seasonal_df_metric_UpscaledEC = seasonal_df_UpscaledEC.apply(lambda x: (np.max(x[3:11]) - np.min(x[3:11])), axis=0)
    ylabel = 'NEE seasonal amplitude\n(µmol m⁻² s⁻¹)'
    reference = 1.0  # reference value for range ratio

elif stat_var == 'cor':
    
    # standardize with minumum NEE
    def scale_minumum (vec):
        return -vec / np.min(vec)
    
    # calculate MAD
    def cal_mad (vec, reference):
        return np.mean(abs(reference[3:11]-vec[3:11])) # only growing season

    # reference
    seasonal_df_reference_mean = seasonal_df_inversions.apply(scale_minumum, axis=0).mean(axis=1)
    seasonal_df_metric_TRENDYv11 = seasonal_df_TRENDY.apply(scale_minumum, axis=0).apply(lambda x: cal_mad(x, seasonal_df_reference_mean), axis=0)
    seasonal_df_metric_inversions = seasonal_df_inversions.apply(scale_minumum, axis=0).apply(lambda x: cal_mad(x, seasonal_df_reference_mean), axis=0)
    seasonal_df_metric_UpscaledEC = seasonal_df_UpscaledEC.apply(scale_minumum, axis=0).apply(lambda x: cal_mad(x, seasonal_df_reference_mean), axis=0)
    ylabel = 'Seasonal cycle shape bias'
    
# plot
fig, ax = plt.subplots(figsize=(4, 3.5))

# Define colors for groups and subgroups
colors = {'Inversions': 'black', 'Upscaled fluxes': '#4c8638', 'TRENDY TBMs': '#d4631d'}

plt.scatter(fitting_df_TRENDYv11[stat_var], seasonal_df_metric_TRENDYv11, edgecolor=colors['TRENDY TBMs'], facecolor='none', label='TRENDY TBMs')
plt.scatter(fitting_df_UpscaledEC[stat_var], seasonal_df_metric_UpscaledEC, edgecolor=colors['Upscaled fluxes'], facecolor='none', label='Upscaled fluxes')
plt.scatter(fitting_df_inversions[stat_var], seasonal_df_metric_inversions, edgecolor=colors['Inversions'], facecolor='none', label='Inversions')

# add regression line
x = np.concatenate([
    fitting_df_TRENDYv11[stat_var].values,
    fitting_df_UpscaledEC[stat_var].values,
    fitting_df_inversions[stat_var].values
])
y = np.concatenate([
    seasonal_df_metric_TRENDYv11.values,
    seasonal_df_metric_UpscaledEC.values,
    seasonal_df_metric_inversions.values
])
m, b = np.polyfit(x, y, 1)
x0 = np.linspace(xlim[0], xlim[1], 100)
plt.plot(x0, m * x0 + b, color='k', linestyle='--', alpha=0.8)
if stat_var in ['mean_bias', 'range_ratio_95_5']:
    y_ref = m * reference + b
    print(f'{ylabel}: {y_ref}, at {stat_var} = {reference}')

    # calculate cell area in m2
    def calculate_area(latitudes, res_x, res_y):
        """Calculate grid cell area in m2."""
        re = 6371220  # Earth radius in meters
        rad = np.pi / 180.0  # Radians per degree
        con = re * rad  # Distance per degree
        clat = np.cos(latitudes * rad)  # Cosine of latitude in radians
        dlon = res_x  # Longitude resolution
        dlat = res_y  # Latitude resolution
        dx = con * dlon * clat  # dx at each latitude
        dy = con * dlat  # dy is constant
        dxdy = dy * dx  # Area of each grid cell
        return dxdy

    res_x = 0.5
    res_y = 0.5
    import xarray as xr
    ABoVE_mask = xr.open_dataset('/resnick/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/above_mask/above_ext.nc')
    latitudes = ABoVE_mask['lat'].values
    area = calculate_area(latitudes, res_x, res_y)  # m2
    area_2d = np.tile(area[:, np.newaxis], (1, ABoVE_mask.dims['lon']))  # make it 2D
    area_da = xr.DataArray(area_2d, coords=[ABoVE_mask['lat'], ABoVE_mask['lon']], dims=['lat', 'lon'])
    area_da = area_da.where(ABoVE_mask['ids'] == 0)
    area_total = area_da.sum().values  # m2
    # convert to TgC
    y_ref_tgC = y_ref * 1e-6 * area_total * 12 * 3600 * 24 * 30 * 8 / 1e12  # µmol m-2 s-1 to TgC
    y_ref_tgC_mo = y_ref * 1e-6 * area_total * 12 * 3600 * 24 * 30 / 1e12  # µmol m-2 s-1 to TgC/mo
    print(f'in TgC: {y_ref_tgC} at {stat_var} = {reference}')
    print(f'in TgC/mo: {y_ref_tgC_mo} at {stat_var} = {reference}')

# 95% confidence band for the regression line (mean prediction)
n = len(x)
x_mean = np.mean(x)
Sxx = np.sum((x - x_mean) ** 2)
resid = y - (m * x + b)
s = np.sqrt(np.sum(resid ** 2) / (n - 2))

yhat = m * x0 + b
se_mean = s * np.sqrt(1 / n + (x0 - x_mean) ** 2 / Sxx)
y_lo = yhat - 1.96 * se_mean
y_hi = yhat + 1.96 * se_mean

plt.fill_between(x0, y_lo, y_hi, color='k', alpha=0.15, linewidth=0)

corr, _ = pearsonr(x, y)

if stat_var == 'cor':
    plt.text(0.05, 0.05, f'Cor: {"{:.2f}".format(corr)}',
         transform=ax.transAxes, ha='left', va='bottom', fontsize=12)
else:
    plt.text(0.95, 0.05, f'Cor: {"{:.2f}".format(corr)}',
         transform=ax.transAxes, ha='right', va='bottom', fontsize=12)

plt.xlabel(xlabel, fontsize=12)
plt.ylabel(ylabel, fontsize=12)
plt.xlim(xlim)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()