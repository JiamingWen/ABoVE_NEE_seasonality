# read hdf5 files from GFED
import h5py
import numpy as np
import pandas as pd

year = 2019
month = 1
monthstr = str(month).zfill(2)

input_dir: str = "/central/groups/carnegie_poc/michalak-lab/data/gfed/v4.1s/"
if (year <= 2016):
    filename = input_dir + f"GFED4.1s_{year}.hdf5"
else:
    filename = input_dir + f"GFED4.1s_{year}_beta.hdf5"

f = h5py.File(filename)
for key in f.keys():
    print(key) #Names of the root level object names in HDF5 file - can be groups or datasets.
    print(type(f[key])) # get the object type: usually group or dataset

f['lat'][()]
f['lon'][()]

key = 'emissions'
group = f[key]
for key in group.keys():
    print(key)

data = group['01']
for key in data.keys():
    print(key)

data_FE = data['C'][()] #convert to np array

###############################################
C_emission = f['emissions'][monthstr]['C'][()] #unit g C m-2 month-1
DM_emission = f['emissions'][monthstr]['DM'][()] #unit kg DM m-2 month-1

# DM_AGRI_frac = f['emissions'][monthstr]['partitioning']['DM_AGRI'][()] #range 0-1
# DM_BORF_frac = f['emissions'][monthstr]['partitioning']['DM_BORF'][()]
# DM_DEFO_frac = f['emissions'][monthstr]['partitioning']['DM_DEFO'][()]
# DM_PEAT_frac = f['emissions'][monthstr]['partitioning']['DM_PEAT'][()]
# DM_SAVA_frac = f['emissions'][monthstr]['partitioning']['DM_SAVA'][()]
# DM_TEMF_frac = f['emissions'][monthstr]['partitioning']['DM_TEMF'][()]

emission_factor = pd.read_table('/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/gfed/v4/GFED4_Emission_Factors.txt', 
                                skiprows=17, header=None, nrows=3, delim_whitespace=True, 
                                names=['C_species','AGRI', 'BORF', 'DEFO', 'PEAT', 'SAVA', 'TEMF'])


CO2_emission = np.zeros(shape=DM_emission.shape)

for LCname in ['AGRI', 'BORF', 'DEFO', 'PEAT', 'SAVA', 'TEMF']:
    LC_DM_frac = f['emissions'][monthstr]['partitioning']['DM_'+LCname][()]
    LC_DM_emission = DM_emission * LC_DM_frac
    LC_CO2_emission = LC_DM_emission * float(emission_factor[LCname][emission_factor['C_species']=='CO2'])
    CO2_emission += LC_CO2_emission