#!/usr/bin/python

"""
This file is adopted from Yoichi's hsplit.py file for inversion.

Go through the netcdf footprint files and create the H slices at target timestamps

Do this by stepping through the observations.  For each obs, read in the
netcdf file.  Then step through the H block timesteps, find the footprint values
that match the timestep, and write any nonzero values to a temporary text H slice file.
Each subsequent observation footprint will append to the H text files.

---
for each obs:
	for each footprint timestep:
		read nonzero footprint
	for each target timestep
		append values to text files, H1, H2, H3, ...

---

This way we read each netcdf file only 1 time.

Requires:
	receptor list - list of footprint file names to use
	footprint files - stilt footprint files
	config file - Configuration file

Output:
	H text files.  There is one file for each timestep.
"""

from __future__ import print_function
import os
import datetime
import shutil
import numpy as np
import pandas as pd
import math

os.chdir('/central/groups/carnegie_poc/jwen2/ABoVE/tmp/src')
import utils # import from an outside script

year = 2014 # 2012 2013 2014

receptor_df = pd.read_csv(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/data/carve_airborne/ABoVE_{year}_carve_airborne_change.csv')
list_footprint_files = receptor_df['footprint_filename'].tolist() #list of the footprint files
footprint_dir = f"/central/groups/carnegie_poc/michalak-lab/nasa-above/data/input/footprints/carve/CARVE_L4_WRF-STILT_Footprint/data/CARVE-{year}-aircraft-footprints-convect/"

config = utils.getConfig(f'/central/groups/carnegie_poc/jwen2/ABoVE/ABoVE_NEE_seasonality/src/config_carve{year}.ini')
t0 = datetime.datetime.now()

# Directory to store H matrix
hdir = config["hdir"]
if os.path.exists(hdir):
	# remove any files in hdir
	shutil.rmtree(hdir)

# create hdir
if not os.path.exists(hdir):
	os.makedirs(hdir)

# Loop through each of the footprint files
for obsnum, line in enumerate(list_footprint_files): #obsnum - index; line - filename

	filename = footprint_dir + line

	t1 = datetime.datetime.now()

	# read the netcdf file for the obs.
	# grid returned is (ntimesteps x lat x lon) for 0.5 degree grid
	grid_3d, griddates, lats, lons = utils.get_footprint(filename)

	# note that lons is 140.25, ..., 179.75, -179.75, ..., 139.75
	# change it to -179.75, ..., 179.75
	if lons[0] == 140.25:
		grid_3d = np.concatenate((grid_3d[:,:,lons < 140], grid_3d[:,:,lons > 140]), axis=2)
		lons = np.concatenate((lons[lons < 140], lons[lons > 140]))
	else:
		print('lon is in different format')

	# Changes shape of grid to 2d ntimesteps x ncells
	tmp = np.meshgrid(np.arange(len(lons)), np.arange(len(lats)))
	lats_id = tmp[1].flatten().astype('i').reshape(1, len(lats)*len(lons))
	lons_id = tmp[0].flatten().astype('i').reshape(1, len(lats)*len(lons))
	coor_array = np.concatenate((lats_id, lons_id), axis=0)
	grid = grid_3d[:, coor_array[0], coor_array[1]]

	'''
	Loop through each of the griddates, find ones with dates in desired range.
	We're going backwards in time, so skip dates after the end date, quit when before start date.
	If found, extract nonzero grid indices, write those to file for correct target timestep
	If target timestep > footprint timestep, then sum up the footprint values that fit in the target timestep window
	e.g. if target timestep is 3 hours, then sum up hours 0, 1, 2 for timestep 1; 3, 4, 5 for timestep 2 ...
	'''
	xx = {}
	for gidx, gd in enumerate(griddates):
		if gd < config["sdate"]: break
		if gd >= config["edate"]: continue

		nmonths = (gd.year - config["sdate"].year) * 12 + gd.month - config["sdate"].month		# months since the starting month
		ntimestep = math.floor(nmonths / config["timestep_in_months"]) + 1  	# timestep number, starting from 1
		nz = grid[gidx].nonzero()[0]			# location of non zero data points
		for cellnum in nz:
			val = grid[gidx, cellnum]
			if val > -3e34:
				# create a dict entry for this timestep and cell number
				if (ntimestep, cellnum) not in xx:
					xx[(ntimestep, cellnum)] = 0

				# add value to this timestep, cellnum
				xx[(ntimestep, cellnum)] += val



	# make a dict that is indexed by timestep only,
	# i.e. contains aggregated nonzero sensitivity of each pixel for each timestep for this observation
	b = {}
	for (timestep, cellnum) in xx:
		if timestep not in b: b[timestep] = []
		lat_id = coor_array[0, cellnum]
		lon_id = coor_array[1, cellnum]
		lat_val = lats[lat_id]
		lon_val = lons[lon_id]
		b[timestep].append((obsnum, cellnum, lat_id, lon_id, lat_val, lon_val, xx[(timestep, cellnum)]))


	# write data to text files,
	for ntimestep in b:
		time_tmp = config["sdate"].year * 12 + config["sdate"].month + ntimestep - 1
		current_year = time_tmp//12
		current_month = time_tmp%12
		tmpfile = "%s/H%d_%d.txt" % (hdir, current_year, current_month)
		f = open(tmpfile, "a")
		for (nobs, cellnum, lat_id, lon_id, lat_val, lon_val, val) in b[ntimestep]:
			f.write("%d %d %d %d %.2f %.2f %15.8e\n" % (nobs, cellnum, lat_id, lon_id, lat_val, lon_val, val))
		f.close()


	t2 = datetime.datetime.now()
	print("Finished obs num ", obsnum, t2-t1, t2-t0)


# # may consider storing them into hdf5 files to reduce size
# # utils.py also have some functions to store the files in other formats, e.g., *.npz - save ~50% space
# start_month = int(config['start_month'].split('-')[1])
# end_month = int(config['end_month'].split('-')[1]) 
# for month in np.arange(start_month,end_month+1):
# 	h_df = pd.read_csv(f"/central/groups/carnegie_poc/jwen2/ABoVE/carve_airborne/h_matrix/h_sparse_matrix/H{year}_{month}.txt",
# 					sep="\s+", index_col=False, header=None,
# 					names=["obs_id", "cell_id", "lat_id","lon_id", "lat", "lon", "val"])
# 	tmpfile = "%s/H%d_%d.npz" % (hdir, year, month)
# 	np.savez(tmpfile, rows=h_df['obs_id'], cols=h_df['cell_id'], vals=h_df['val'], shape=(26128, 86400))