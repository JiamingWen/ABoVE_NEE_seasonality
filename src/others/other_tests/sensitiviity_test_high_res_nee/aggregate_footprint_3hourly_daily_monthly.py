'''aggregate footprints from 3-hourly to daily and monthly'''

from functions import get_campaign_info
import utils
from datetime import datetime, timedelta
import os
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

year = 2012 # 2012 2013 2014 2017

campaign_name = get_campaign_info(year)[2]
config = utils.getConfig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/config/config_{campaign_name}{year}_monthly.ini')

# read observations
receptor_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
n_receptor = receptor_df.shape[0]

dir_3hourly = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/3hourly'
dir_daily = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/daily'
dir_monthly = f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/monthly'
if not os.path.exists(dir_daily):
    os.makedirs(dir_daily)
if not os.path.exists(dir_monthly):
    os.makedirs(dir_monthly)

h_filelist_3hourly = [f for f in os.listdir(dir_3hourly) if f.endswith('.txt')]

for month in range(1, 13):
    print(month)

    h_filelist_monthly = [f for f in h_filelist_3hourly if f.startswith(f'H{year}_{month}_')]
    h_matrix_monthly = None

    # daily
    for day in range(1, 32):
        h_filelist_daily = [f for f in h_filelist_monthly if f.startswith(f'H{year}_{month}_{day}_')]
        
        if len(h_filelist_daily) > 0:
            h_matrix_daily = None
            for h_file in h_filelist_daily:
                h_df = pd.read_csv(os.path.join(dir_3hourly, h_file), sep="\s+", index_col=False, header=None,
                                   names=["obs_id", "cell_id", "lat_id", "lon_id", "lat", "lon", "val"])
                # Create sparse matrix directly
                n_cell = 720 * 120
                h_matrix_3hourly = csr_matrix((h_df.val, (h_df.obs_id, h_df.cell_id)), shape=(n_receptor, n_cell))
                
                # Concatenate sparse matrices
                if h_matrix_daily is None:
                    h_matrix_daily = h_matrix_3hourly
                else:
                    h_matrix_daily += h_matrix_3hourly
            
            # Save daily H matrix
            output_file = os.path.join(dir_daily, f'H{year}_{month}_{day}.txt')
            with open(output_file, "w") as f:
                coo = h_matrix_daily.tocoo()
                for obs_id, cell_id, val in zip(coo.row, coo.col, coo.data):
                    lat_id = cell_id // 720
                    lon_id = cell_id % 720
                    lat_val = 30.25 + lat_id * 0.5
                    lon_val = -179.75 + lon_id * 0.5
                    f.write(f"{obs_id} {cell_id} {lat_id} {lon_id} {lat_val:.2f} {lon_val:.2f} {val}\n")
            
            # # add to monthly
            # if h_matrix_monthly is None:
            #     h_matrix_monthly = h_matrix_daily
            # else:
            #     h_matrix_monthly += h_matrix_daily
            
    # # Save monthly H matrix
    # if h_matrix_monthly is not None:
    #     output_file = os.path.join(dir_monthly, f'H{year}_{month}_agg.txt')
    #     with open(output_file, "w") as f:
    #         coo = h_matrix_monthly.tocoo()
    #         for obs_id, cell_id, val in zip(coo.row, coo.col, coo.data):
    #             lat_id = cell_id // 720
    #             lon_id = cell_id % 720
    #             lat_val = 30.25 + lat_id * 0.5
    #             lon_val = -179.75 + lon_id * 0.5
    #             f.write(f"{obs_id} {cell_id} {lat_id} {lon_id} {lat_val:.2f} {lon_val:.2f} {val}\n")


# '''check monthly H matrix with previous results'''
# campaign_name = 'carve'
# year = 2012
# month = 5
# receptor_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/atm_obs/ABoVE_{year}_{campaign_name}_airborne_change.csv')
# n_receptor = receptor_df.shape[0]
# n_cell = 720 * 120

# # read stored H sparse matrix - previous
# h_df1 = pd.read_csv(
#     f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/monthly/H{year}_{month}.txt",
#     sep="\s+", index_col=False, header=None,
#     names=["obs_id", "cell_id", "lat_id", "lon_id", "lat", "lon", "val"]
# )
# h_matrix1 = csr_matrix((h_df1.val, (h_df1.obs_id, h_df1.cell_id)), shape=(n_receptor, n_cell))

# # read new H sparse matrix - aggregated from 3-hourly
# h_df2 = pd.read_csv(
#     f"/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/{campaign_name}_airborne/h_matrix/h_sparse_matrix/{year}/monthly/H{year}_{month}_agg.txt",
#     sep="\s+", index_col=False, header=None,
#     names=["obs_id", "cell_id", "lat_id", "lon_id", "lat", "lon", "val"]
# )
# h_matrix2 = csr_matrix((h_df2.val, (h_df2.obs_id, h_df2.cell_id)), shape=(n_receptor, n_cell))

# # sort h_df1 and h_df2 by obs_id and cell_id
# h_df1_sort = h_df1.sort_values(by=['obs_id', 'cell_id']).reset_index(drop=True)
# h_df2_sort = h_df2.sort_values(by=['obs_id', 'cell_id']).reset_index(drop=True)

# np.sum(abs(h_df1_sort.obs_id - h_df2_sort.obs_id)) # 0
# np.sum(abs(h_df1_sort.cell_id - h_df2_sort.cell_id)) # 0
# np.sum(abs(h_df1_sort.val - h_df2_sort.val)) # not equal to 0

# # select rows with non-zero difference
# diff = abs(h_df1_sort.val - h_df2_sort.val)
# non_zero_diff = np.where(diff > 0.000001)[0]
# print(non_zero_diff) # the difference is very small, probably due to precision error

# diff_percent = diff / abs(h_df1_sort.val)
# non_zero_diff_percent = np.where(diff_percent > 0.0001)[0]
# print(non_zero_diff_percent)

# # check the values
# for i in non_zero_diff:
#     print(f"obs_id: {h_df1_sort.obs_id[i]}, cell_id: {h_df1_sort.cell_id[i]}, val1: {h_df1_sort.val[i]}, val2: {h_df2_sort.val[i]}")