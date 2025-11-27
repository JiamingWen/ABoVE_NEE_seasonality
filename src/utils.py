"""
functions used for generating h matrix
"""

import sys
import os
from configobj import ConfigObj
import datetime
import netCDF4
import numpy


#############################################################################
def getConfig(configfile="config.ini"):
	""" Read in the configuration file, create the config dict, and
	calculate additional entries.
	Uses ConfigObj module. https://github.com/DiffSK/configobj
	"""

	if not os.path.exists(configfile):
		sys.exit("Error: config file '%s' not found." % configfile)


	config = ConfigObj(configfile, unrepr=True)

	# Get start and end dates
	if "timestep_in_months" in config and config["timestep_in_months"] == 1: # monthly
		(syr, smon, sday) = config["start_month"].split("-") + ['01']
		(eyr, emon, eday) = config["end_month"].split("-") + ['01']
		config["sdate"] = datetime.datetime(int(syr), int(smon), int(sday), 0, 0)
		config["edate"] = datetime.datetime(int(eyr), int(emon)+1, int(eday), 0, 0)
		config["nmonths"] = (config["edate"].year - config["sdate"].year) * 12 + config["edate"].month - config["sdate"].month
		config["ntimesteps"] = int(config["nmonths"] / config["timestep_in_months"])

	if "hrsperstep" in config and config["hrsperstep"] == 3: # 3-hourly
		(syr, smon, sday) = config["start_date"].split("-")
		(eyr, emon, eday) = config["end_date"].split("-")
		config["sdate"] = datetime.datetime(int(syr), int(smon), int(sday))
		config["edate"] = datetime.datetime(int(eyr), int(emon), int(eday))
		config["oneday"] = datetime.timedelta(days=1)
		config["ndays"] = (config["edate"]-config["sdate"] + config["oneday"]).days

		config['steps_per_day'] = 24.0/config["hrsperstep"]
		config["ntimesteps"] = int(config["ndays"] * config["steps_per_day"])
		config["timestep"] = datetime.timedelta(hours=config["hrsperstep"])

	config["nlatgrids"] = (config["north"] - config["south"]) / config["lat_resolution"]
	config["nlongrids"] = (config["east"] - config["west"]) / config["lon_resolution"]
	config["ncells"] = config["nlatgrids"] * config["nlongrids"]

	return config


def get_footprint(filename):
	"""
	Return from the netcdf file 'filename' the footprint grid.
	This the the netcdf variable 'foot1' from the stilt netcdf footprint files.
	"""

	try:
		ds = netCDF4.Dataset(filename)
	except IOError as e:
		print("Error trying to read netcdf file %s. %s" % (filename, e), file=sys.stderr)
		return None, None

	g = ds.variables['foot1']
	grid = g[:]
	lats = ds.variables['foot1lat'][:]
	lons = ds.variables['foot1lon'][:]
	dy = lats[1] - lats[0]
	dx = lons[1] - lons[0]

	s = list(g.dimensions)
	n1 = s.index("foot1lat")
	n2 = s.index("foot1lon")

	# Convert the foot1date array from days since jan 1, 2000 to actual datetime
	f1 = ds.variables['foot1date']
	dates = netCDF4.num2date(f1[:], f1.units)

	ds.close()

	# if 'foot1lon' dimension comes before 'foot1lat', swap the dimensions
	# some netcdf files store the footprint as (nstep x lon x lat).
	# We need (nstep x lat x lon).  Switch lat, lon around here
	if n2 < n1:
		print('Footprint file lat/lon reversed')
		b = numpy.empty((grid.shape[0], grid.shape[2], grid.shape[1]))
		nrows = grid.shape[0]
		for i in range(nrows):
			a = grid[i]
			b[i] = a.T

		grid = b

	# check if spatial resolution is 0.5 degree
	if dx != 0.5:
		print('The resolution of footprint file is not 0.5 degree')

	return grid, dates, lats, lons


def write_sparse(filename, matr):
	"""
	Write a sparse matrix to 'filename',
	using the numpy savez format.
	"""

	m = numpy.array(matr)

	# get indices of non-zero values
	a = m.nonzero()
	rows = numpy.array(a[0])
	cols = numpy.array(a[1])
	vals = numpy.array(matr[(rows, cols)])

	numpy.savez(filename, rows=rows, cols=cols, vals=vals, shape=matr.shape)