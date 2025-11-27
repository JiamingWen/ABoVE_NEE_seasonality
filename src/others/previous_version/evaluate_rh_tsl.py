'''evaluate the functional relationship between Rh and tsl'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import os
os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/src')
from functions import get_campaign_info, read_TRENDYv11, read_TRENDYv11_cPool

high_model_subset = ['CLASSIC', 'ISBA-CTRIP']
low_model_subset = ['JSBACH', 'JULES'] # can add ISAM later if the data becomes available


'''some anciliary datasets'''
# regional mask and land cover table
cell_id_table = pd.read_csv('/central/groups/carnegie_poc/jwen2/ABoVE/cell_id_table/cell_id_table.csv')
region_mask = np.where(cell_id_table['ABoVE'] == 0)[0]

'''extract simulated Rh and tsl pairs for each pixel, each month and year'''
for year in [2012, 2013, 2014, 2017]:

    start_month, end_month, campaign_name = get_campaign_info(year)

    # create dir
    dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/rh_tsl/"
    if not os.path.exists(dir0):
        os.makedirs(dir0)

    for model_name in high_model_subset+low_model_subset:

        result_df = pd.DataFrame()
        
        for month in np.arange(4, 12):

            rh_vec = read_TRENDYv11(model_name, 'rh', year, month).values.flatten()
            rh_vec_unit = rh_vec*1000/12*1e6 #convert unit to μmol m-2 s-1
            rh = np.nan_to_num(rh_vec_unit, nan=0)

            tsl = read_TRENDYv11(model_name, 'tsl', year, month).values.flatten()

            # carbon pools at monthly resolution
            if model_name == 'ISBA-CTRIP':
                cLitter_month = read_TRENDYv11(model_name, 'cLitter', year, month).values.flatten()
                cSoil_month = read_TRENDYv11(model_name, 'cSoil', year, month).values.flatten()
            else:
                cLitter_month = np.full(len(tsl), np.nan)
                cSoil_month = np.full(len(tsl), np.nan)

            # carbon pools at annual resolution
            cSoil_annual = read_TRENDYv11_cPool(model_name, 'cSoil', year).values.flatten()
            if model_name in ['CLASSIC', 'ISBA-CTRIP', 'JSBACH']:
                cLitter_annual = read_TRENDYv11_cPool(model_name, 'cLitter', year).values.flatten()
            else:
                cLitter_annual = np.full(len(tsl), 0.0)

            result_df_month = pd.DataFrame({'cell_id': cell_id_table.loc[region_mask, 'cell_id'],
                                            'lc': cell_id_table.loc[region_mask, 'lc'],
                                            'month': np.ones(len(region_mask))*month,
                                            'rh': rh[region_mask],
                                            'tsl': tsl[region_mask],
                                            'cLitter_month': cLitter_month[region_mask],
                                            'cSoil_month': cSoil_month[region_mask],
                                            'cLitter_annual': cLitter_annual[region_mask],
                                            'cSoil_annual': cSoil_annual[region_mask]
                                            })
        
            result_df = pd.concat((result_df, result_df_month))

        result_df.to_csv(f'{dir0}/rh_tsl_{model_name}_{year}.csv', encoding='utf-8', index=False)
        # plt.scatter(result_df['tsl'], result_df['rh'])



'''compare carbon pools at monthly and annual resolution'''
model_name = 'ISBA-CTRIP'
data = pd.DataFrame()
for year in [2012, 2013, 2014, 2017]:

    start_month, end_month, campaign_name = get_campaign_info(year)
    dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/rh_tsl/"
    data_year = pd.read_csv(f'{dir0}/rh_tsl_{model_name}_{year}.csv')
    data = pd.concat((data, data_year))

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(data['cLitter_month'], data['cLitter_annual'])
plt.xlabel('cLitter (monthly)')
plt.ylabel('cLitter (annual)')

plt.subplot(1, 2, 2)
plt.scatter(data['cSoil_month'], data['cSoil_annual'])
plt.xlabel('cSoil (monthly)')
plt.ylabel('cSoil (annual)')

plt.show()


'''compare carbon pools between different models'''
cPool_df = pd.DataFrame()
var_name = 'cLitter_annual' # cLitter_annual cSoil_annual
for model_name in high_model_subset+low_model_subset:

    data = pd.DataFrame()
    for year in [2012, 2013, 2014, 2017]:

        start_month, end_month, campaign_name = get_campaign_info(year)
        dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/rh_tsl/"
        data_year = pd.read_csv(f'{dir0}/rh_tsl_{model_name}_{year}.csv')
        data = pd.concat((data, data_year))
    print(data.shape)

    cPool_df = pd.concat((cPool_df, pd.DataFrame({model_name: data[var_name]})), axis=1)
    print(cPool_df)

cPool_df.corr()


'''plot spatial variations of cLitter and cSoil'''
import cartopy.crs as ccrs
var_name = 'cLitter' #cSoil cLitter

for year in [2012, 2013, 2014, 2017]:

    start_month, end_month, campaign_name = get_campaign_info(year)

    for model_name in high_model_subset+low_model_subset:

        if var_name == 'cLitter' and model_name == 'JULES':
            continue
        else:
            cPool_annual = read_TRENDYv11_cPool(model_name, var_name, year)

            fig = plt.figure(figsize=(6,3))
            ccrs_plot = ccrs.PlateCarree()
            region_extent = [-170, -80, 48, 80]; subtitle_loc = [-165, 76] # core+extended
            ax = plt.axes(projection=ccrs_plot)
            ax.set_extent(region_extent, ccrs_plot)
            gl = ax.gridlines(draw_labels=True) #linewidth=2, color='gray', alpha=0.5, linestyle='--'
            ax.coastlines()
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'fontsize': 12}
            gl.ylabel_style = {'fontsize': 12}
            ax.set_aspect(2)
            plt.title(model_name+' '+str(year))

            lons = cPool_annual["longitude"].values
            lats = cPool_annual["latitude"].values
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            cax = fig.add_axes([0.8, 0.1, 0.02, 0.7])
            cp = ax.pcolormesh(lon_grid, lat_grid, cPool_annual.squeeze(dim='time', drop=True)) #OrRd #, vmin=0, vmax=100, cmap='Purples'
            cb = fig.colorbar(cp, cax=cax, orientation="vertical")
            cb.ax.tick_params(labelsize=15)
            cb.set_label(
                f"{var_name} (kg/m2)",
                fontsize=18,
            )
            plt.show()




'''scatterplot for all data'''
for model_name in high_model_subset+low_model_subset:

    data = pd.DataFrame()
    for year in [2012, 2013, 2014, 2017]:

        start_month, end_month, campaign_name = get_campaign_info(year)
        dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/rh_tsl/"
        data_year = pd.read_csv(f'{dir0}/rh_tsl_{model_name}_{year}.csv')
        data = pd.concat((data, data_year))

    fig, ax = plt.subplots(figsize=(4, 4))
    # plt.scatter(data['tsl'], data['rh'])

    # only select forest pixels
    # data = data.loc[data['lc'] == 5]
    print(data.shape)

    
    data_filtered = data[['rh', 'tsl', 'cLitter_annual', 'cSoil_annual']].dropna()
    x = data_filtered['tsl'] - 273.15
    # y = data_filtered['rh']
    # y = data_filtered['rh'] / data_filtered['cSoil_annual']
    y = data_filtered['rh'] / (data_filtered['cSoil_annual'] + data_filtered['cLitter_annual'])
    tmp = pd.DataFrame({'x':x,'y':y})
    tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
    tmp.dropna(inplace=True)
    xy = np.vstack([tmp['x'], tmp['y']])
    z = gaussian_kde(xy)(xy)
    ax.scatter(tmp['x'], tmp['y'], c=z, s=100)

    # fit curve
    from scipy.optimize import curve_fit
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    
    popt, pcov = curve_fit(exp_func, tmp['x'], tmp['y'])
    grid = np.arange(min(tmp['x']), max(tmp['x']), 0.1)
    plt.plot(grid, exp_func(grid, *popt), 'r-')


    # adjust plot settings
    plt.title(model_name)
    # plt.xlim(240,300)
    plt.xlim(-30,30)
    plt.xlabel('Soil temperature')
    # plt.ylabel('Rh')
    plt.ylabel('Rh / (cSoil+cLitter)')
    plt.show()


'''every year, colored in different months'''
for model_name in high_model_subset+low_model_subset:

    # data = pd.DataFrame()
    for year in [2012, 2013, 2014, 2017]:

        start_month, end_month, campaign_name = get_campaign_info(year)
        dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/rh_tsl/"
        data_year = pd.read_csv(f'{dir0}/rh_tsl_{model_name}_{year}.csv')

        fig, ax = plt.subplots(figsize=(4, 4))
        for month in np.arange(4,12):
            data_month = data_year.loc[data_year['month'] == month]
            plt.scatter(data_month['tsl']-273.15, data_month['rh'], label=month)

        plt.title(model_name)
        # plt.xlim(240,300)
        plt.xlim(-30,30)
        plt.xlabel('Soil temperature')
        plt.ylabel('Rh')
        plt.legend()
        plt.show()


'''plot for different months and fit with regression'''
for model_name in high_model_subset+low_model_subset:

    data = pd.DataFrame()
    for year in [2012, 2013, 2014, 2017]:

        start_month, end_month, campaign_name = get_campaign_info(year)
        dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/rh_tsl/"
        data_year = pd.read_csv(f'{dir0}/rh_tsl_{model_name}_{year}.csv')
        data_year = data_year[[column_name for column_name in data_year.columns if column_name not in ['cLitter_month', 'cSoil_month']]]
        data = pd.concat((data, data_year))

    fig, ax = plt.subplots(2,4, figsize=(15, 7))
    for month in np.arange(4,12):
        
        ax1 = plt.subplot(2, 4, month-3)
        data_month = data.loc[data['month'] == month]
        Tsoil = data_month['tsl'] - 273.15
        Rh = data_month['rh']
        cSoil = data_month['cSoil_annual']
        cLitter = data_month['cLitter_annual']

        tmp = pd.DataFrame({'Tsoil':Tsoil, 'Rh':Rh, 'cSoil':cSoil, 'cLitter':cLitter})
        tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
        tmp.dropna(inplace=True)
        # xy = np.vstack([tmp['Tsoil'], tmp['Rh']])
        # z = gaussian_kde(xy)(xy)
        # ax1.scatter(tmp['Tsoil'], tmp['Rh'], c=z, s=50)

        # fit curve
        from scipy.optimize import curve_fit
        def exp_func(X, a, b, c, d):
            Tsoil, cSoil, cLitter = X
            return cSoil * a * np.exp(b * Tsoil) + cLitter * c * np.exp(d * Tsoil)
        
        popt, pcov = curve_fit(exp_func, (tmp['Tsoil'], tmp['cSoil'], tmp['cLitter']), tmp['Rh'])
        y_hat = exp_func((tmp['Tsoil'], tmp['cSoil'], tmp['cLitter']), *popt)
        xy = np.vstack([tmp['Rh'], y_hat])
        z = gaussian_kde(xy)(xy)
        ax1.scatter(tmp['Rh'], y_hat, c=z, s=50)

        popt_round = ["{:.3f}".format(para) for para in popt]
        plt.text(0.5, 4.7, f'a={popt_round[0]}')
        plt.text(0.5, 4.4, f'b={popt_round[1]}')
        plt.text(0.5, 4.1, f'c={popt_round[2]}')
        plt.text(0.5, 3.8, f'd={popt_round[3]}')
        plt.title(model_name+'-'+str(month))
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.xlabel('Rh fitted')
        plt.ylabel('Rh')
    
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()




'''all years, colored in different months, fitted with regression'''
for model_name in high_model_subset+low_model_subset:

    data = pd.DataFrame()
    for year in [2012, 2013, 2014, 2017]:

        start_month, end_month, campaign_name = get_campaign_info(year)
        dir0 = f"/central/groups/carnegie_poc/jwen2/ABoVE/{campaign_name}_airborne/rh_tsl/"
        data_year = pd.read_csv(f'{dir0}/rh_tsl_{model_name}_{year}.csv')
        data = pd.concat((data, data_year))

    fig, ax = plt.subplots(figsize=(4, 4))

    # only select forest pixels
    # data = data.loc[data['lc'] == 5]
    print(data.shape)

    import statsmodels.api as sm
    from scipy.interpolate import interp1d

    for month in np.arange(4,12):
        data_month = data.loc[data['month'] == month]
        # x = data_month['tsl'] - 273.15
        # y = data_month['rh'] / data_month['cSoil_annual']

        # lowess = sm.nonparametric.lowess(y, x, frac=.3)
        # lowess_x = list(zip(*lowess))[0]
        # lowess_y = list(zip(*lowess))[1]
        # f = interp1d(lowess_x, lowess_y, bounds_error=False)
        
        # grid = np.arange(min(x), max(x), 0.1)
        # plt.plot(grid, f(grid), '-', label=month)

        x = data_month['tsl'] - 273.15
        y = data_month['rh'] / (data_month['cSoil_annual']+ data_month['cLitter_annual']) # 

        tmp = pd.DataFrame({'x':x,'y':y})
        tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
        tmp.dropna(inplace=True)
        # plt.scatter(tmp['x'], tmp['y'])

        # fit curve
        from scipy.optimize import curve_fit
        def exp_func(x, a, b):
            return a * np.exp(b * x)
        
        popt, pcov = curve_fit(exp_func, tmp['x'], tmp['y'])
        grid = np.arange(min(tmp['x']), max(tmp['x']), 0.1)
        plt.plot(grid, exp_func(grid, *popt), '-', label=month)

    plt.title(model_name)
    plt.xlim(-30,30)
    plt.ylim(0,0.4)
    plt.xlabel('Soil temperature')
    plt.ylabel('Rh / (cSoil+cLitter)')
    plt.legend()
    plt.show()


