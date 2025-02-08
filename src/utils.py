
"""
Various lpdm utility routines that are used by multiple scripts.
"""

from __future__ import print_function

import sys
import os

import datetime
import struct
import math
import numpy
from scipy.sparse import csr_matrix

import netCDF4
from configobj import ConfigObj


#############################################################################
def getConfig(configfile="config.ini"):
	""" Read in the configuration file, create the config dict, and
	calculate additional entries.
	Uses ConfigObj module. https://github.com/DiffSK/configobj
	"""

	if not os.path.exists(configfile):
		sys.exit("Error: config file '%s' not found." % configfile)


	config = ConfigObj(configfile, unrepr=True)

	# Get start and end dates for the inversion
	(syr, smon, sday) = config["start_month"].split("-") + ['01']
	(eyr, emon, eday) = config["end_month"].split("-") + ['01']
	config["sdate"] = datetime.datetime(int(syr), int(smon), int(sday), 0, 0)
	config["edate"] = datetime.datetime(int(eyr), int(emon)+1, int(eday), 0, 0)
	config["nmonths"] = (config["edate"].year - config["sdate"].year) * 12 + config["edate"].month - config["sdate"].month

	config["ntimesteps"] = int(config["nmonths"] / config["timestep_in_months"])

	config["nlatgrids"] = (config["north"] - config["south"]) / config["lat_resolution"]
	config["nlongrids"] = (config["east"] - config["west"]) / config["lon_resolution"]
	config["ncells"] = config["nlatgrids"] * config["nlongrids"]

	return config


####################################################
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

#######################################################
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

# #########################################################################
# def getCellLatLon(config, cellnum):
# 	""" Given a land cell number, return its latitude and longitude """

# 	if cellnum < 0 or cellnum >= len(config["lats"]):
# 		print("In getCellLatLon, cellnum out of range (%d). Should be >0 and < %d" % (cellnum, len(config["lats"])), file=sys.stderr)
# 		return -999, -999

# 	lat = config["lats"][cellnum]
# 	lon = config["lons"][cellnum]

# 	return lat, lon

# #########################################################################
# def getCellNum(config, lat, lon):
# 	""" Given a latitidue and longitude, find the corresponding land cell number """

# 	for n, (lt, ln) in enumerate(zip(config["lats"], config["lons"])):
# 		if lat == lt and lon == ln:
# 			return n

# 	return -1


# #########################################################################
# def binread_sparse(binfile, getSparse=True):
# 	"""
# 	Read a sparse binary file for H strips in vineet's method,
# 	return a dense matrix if getSparse==False or
# 	return a sparse matrix if getSparse==True (default)
# 	"""

# 	try:
# 		f = open(binfile, "rb")
# 	except IOError as e:
# 		sys.exit("In binread_sparse: Can't open %s for reading." % binfile)

# 	data = numpy.fromfile(f, 'i4', count=1)
# 	nrows = data[0]
# 	rows = numpy.fromfile(f, 'i4', count=nrows)
# 	cols = numpy.fromfile(f, 'i4', count=nrows)
# 	vals = numpy.fromfile(f, 'f8', count=nrows)
# 	f.close()

# 	# files have 1 based indexing for fortran,
# 	# subtract 1 from rows and cols to get 0 based indexing for python
# 	rows = rows - 1
# 	cols = cols - 1

# 	if getSparse:
# 		m = csr_matrix((vals, (rows, cols)))

# 	else:
# 		nr = rows[-1]
# 		nc = cols[-1]

# 		m = numpy.zeros([nr+1, nc+1])
# 		m[rows, cols] = vals

# 	return m

# #########################################################################
# def binwrite_sparse(binfile, matr):
# 	"""
# 	Take a dense matrix 'matr', and write out a sparse representation
# 	in binary format compatible with vineet's method
# 	This is mainly for writing h strips
# 	"""

# 	try:
# 		f = open(binfile, "wb")
# 	except IOError as e:
# 		print("Can't open %s for writing" % binfile, file=sys.stderr)
# 		return


# 	# get indices of non-zero values
# 	a = matr.nonzero()
# 	rows = numpy.array(a[0])
# 	cols = numpy.array(a[1])
# 	vals = numpy.array(matr[(rows, cols)])
# 	nrows = rows.size

# 	# if last row,col point is zero, include it in file so
# 	# that we know the dimension of full matrix
# 	if matr[-1, -1] == 0:
# 		d = matr.shape
# 		if nrows == 0:
# 			rows = numpy.array([d[0]-1])
# 			cols = numpy.array([d[1]-1])
# 			vals = numpy.array([0])
# 		else:
# 			rows = numpy.append(rows, d[0]-1)
# 			cols = numpy.append(cols, d[1]-1)
# 			vals = numpy.append(vals, 0)
# 		nrows += 1


# 	# header, nrows by ncols (ncols always 3)
# 	data = struct.pack("i", nrows)
# 	f.write(data)

# 	# add one to row and col indices to make compatible with fortran
# 	rows = rows.astype(numpy.int32) + 1
# 	cols = cols.astype(numpy.int32) + 1
# 	rows.tofile(f)
# 	cols.tofile(f)
# 	vals.tofile(f)

# 	f.close()






# #######################################################
# def makebinsparse(filename, a, nrows, ncols):
# 	""" save a sparse matrix in binary format compatible with FORTRAN
# 	This version is when we already have the row/column/value indices in
# 	seperate arrays.

# 	Input:
# 		a - record array of (row indices, column indices, non-zero values)
# 		nrows - number of rows in full matrix
# 		ncols - number of columns in full matrix

# 	Note: The dtype of the record array is
# 		dtype=[('obs', numpy.int32), ('cell', numpy.int32), ('val', float)])
# 	"""

# 	# append a 0 value at last matrix cell so we know the full size of matrix
# 	rows = numpy.append(a['obs'], nrows-1)
# 	cols = numpy.append(a['cell'], ncols-1)
# 	vals = numpy.append(a['val'], 0.0)

# 	nr = len(rows)
# 	f = open(filename, "wb")
# 	data = struct.pack("i", nr)
# 	f.write(data)

# 	# add one to row and col indices to make compatible with fortran
# 	rows = rows.astype(numpy.int32) + 1
# 	cols = cols.astype(numpy.int32) + 1

# 	rows.tofile(f)
# 	cols.tofile(f)
# 	vals.tofile(f)

# 	f.close()


# #############################################################
# def makeTemporalCovariance(temporal_cl, config):
# 	""" Compute temporal covariance matrix
# 	Algorithm from vineet's fortran 90 code
# 	subroutine temporal_covariance in
# 	library_inverse.f90

# 	Within day correlations will be 0, day to day
# 	correlations will be > 0.
# 	So e.g. 8 hour timestep, every 3rd value > 0,
# 	correlations will look like

# 	1 0 0 n 0 0 n 0 0
# 	0 1 0 0 n 0 0 n 0
# 	0 0 1 0 0 n 0 0 n
# 	n 0 0 1 0 0 n 0 0
# 	0 n 0 0 1 0 0 n 0
# 	0 0 n 0 0 1 0 0 n
# 	n 0 0 n 0 0 1 0 0
# 	0 n 0 0 n 0 0 1 0
# 	0 0 n 0 0 n 0 0 1

# 	where 0 < n < 1

# 	temporl_cl is correlation length value, whose units are number of days

# 	"""

# 	withinday = int(config["steps_per_day"])
# 	ntimesteps = config["ntimesteps"]
# 	ndays = config["ndays"]  # total number of days in inversion


# 	# if one or more timesteps per day, build array with 0 for within day correlation
# 	if withinday >= 1:

# 		# create an array size = ndays, fill out with day number
# 		t_dist = numpy.empty((ndays, ndays))
# 		for i in range(ndays):
# 			t_dist[i, i:] = numpy.arange(ndays-i)
# 			t_dist[i, 0:i] = numpy.arange(i, 0, -1)

# 		# apply correlation length
# 		t_dist = numpy.exp(-t_dist/temporal_cl)

# 		# expand to ntimesteps x ntimesteps, with cells within same day = 0
# 		temp_cov = numpy.kron(t_dist, numpy.identity(withinday))

# 	else:

# 		# create an array size = ntimesteps, fill out with timestep number
# 		t_dist = numpy.empty((ntimesteps, ntimesteps))
# 		for i in range(ntimesteps):
# 			t_dist[i, i:] = numpy.arange(ntimesteps-i)
# 			t_dist[i, 0:i] = numpy.arange(i, 0, -1)

# 		# convert time step number to days
# 		days_per_step = 1.0 / withinday
# 		t_dist = t_dist*days_per_step

# 		# apply correlation length
# 		temp_cov = numpy.exp(-t_dist/temporal_cl)

# 	return temp_cov


# ###############################################################
# def make_ncdf(filename, name, shat, config):
# 	"""
# 	Convert a numpy array of shape #timesteps x #cells
# 	(#timesteps x (latxlon))
# 	to a netcdf file.

# 	Input:
# 		filename - netcdf output file name
# 		name - name of variable
# 		shat - numpy data array
# 		config - configuration dict

# 	"""

# 	if os.path.exists(config["landmask_file"]):
# 		landmask = numpy.load(config["landmask_file"])


# 	# convert shat dimensions #timesteps x #cells to
# 	# #timesteps x lats x lons for full domain

# 	nlat = config["nlatgrids"]
# 	nlon = config["nlongrids"]
# 	ntimesteps = config["ntimesteps"]


# 	dates = []
# 	date = config["sdate"]
# 	while date < config["edate"]:
# 		dates.append(date)
# 		# dates.append(date + config["timestep"]/2)  # set dates at middle of time step interval
# 		date = date + config["timestep"]



# 	grid = numpy.zeros((ntimesteps, nlat, nlon))
# 	print("grid shape is ", grid.shape)
# 	if os.path.exists(config["landmask_file"]) and not config["isbound"]:
# 		for i in range(ntimesteps):
# 			a = shat[i]
# 			for num, val in enumerate(a):
# 				latidx = landmask[0, num]
# 				lonidx = landmask[1, num]
# 				grid[i, latidx, lonidx] = val

# 	else:
# 		for i in range(ntimesteps):
# 			a = shat[i]
# 			for num, val in enumerate(a):
# 				lat, lon = getCellLatLon(config, num)
# 				latidx = int((lat - config["south"])/config["lat_resolution"])
# 				lonidx = int((lon - config["west"])/config["lon_resolution"])
# 				grid[i, latidx, lonidx] = val

# 	#numpy.save("shat_grid", grid)
# 	#sys.exit()



# #	filename = fluxfile.replace(".npy", ".nc")
# 	ds = netCDF4.Dataset(filename, 'w', format='NETCDF4')

# 	nchar = 500

# 	ds.createDimension('single', 1)
# 	ds.createDimension('nchar', nchar)
# 	ds.createDimension('nlat', nlat)
# 	ds.createDimension('nlon', nlon)
# 	ds.createDimension('ndates', ntimesteps)


# 	#----------
# 	lats = ds.createVariable('lat', 'double', ('nlat'))
# 	lats.units = "degrees_north"
# 	lats.long_name = "latitude"
# 	lats.description = "latitude of center of cells"
# 	latmin = config["south"]
# 	latmax = config["north"]
# 	lats[:] = numpy.arange(latmin, latmax, config["lat_resolution"]) + config["lat_resolution"]/2.0

# 	#----------
# 	latdelta = ds.createVariable('lat_delta', 'double', ('single'))
# 	latdelta.units = "degrees"
# 	latdelta.long_name = "size of cell latitude in degrees"
# 	latdelta[:] = config["lat_resolution"]


# 	#----------
# 	lons = ds.createVariable('lon', 'double', ('nlon'))
# 	lons.units = "degrees_east"
# 	lons.long_name = "longitude"
# 	lons.description = "longitude of center of cells"
# 	lonmin = config["west"]
# 	lonmax = config["east"]
# 	lons[:] = numpy.arange(lonmin, lonmax, config["lon_resolution"]) + config["lon_resolution"]/2.0

# 	#----------
# 	londelta = ds.createVariable('lon_delta', 'double', ('single'))
# 	londelta.units = "degrees"
# 	londelta.long_name = "size of cell longitude in degrees"
# 	londelta[:] = config["lon_resolution"]

# 	#----------
# 	foot1 = ds.createVariable(name, 'float', ('ndates', 'nlat', 'nlon'), fill_value=-1.e+34)
# 	foot1.units = "micromol m-2 s-1"
# 	foot1.long_name = name + " output"

# 	foot1[:] = grid


# 	#----------
# 	d = ds.createVariable('dates', 'float', 'ndates', fill_value=-1.e+34)
# 	d.units = "days since 2000-01-01 00:00:00 UTC"
# 	d.long_name = "dates"

# 	basedate = datetime.datetime(2000, 1, 1)
# 	date = []
# 	for dt in dates:
# 		x = dt - basedate
# 		diff = x.days + x.seconds/86400.0
# 		date.append(diff)

# 	d[:] = numpy.array(date)

# 	ds.close()

# ##########################################################
# def convolve(prior, config, hdir=None):
# 	""" Compute H*prior """

# 	hs = None
# 	timesteps = config["ntimesteps"]

# 	for i in range(timesteps):

# 		s = prior[i]            # prior at this timestep

# 		h = read_sparse_h(i, config, hdir=hdir)
# #		print(h.shape)

# 		if hs is None:
# 			num_obs = h.shape[0]
# 			hs = numpy.zeros(num_obs)


# #               hs = hs + numpy.ravel((h * s))  # the ravel changes nrows x 1 column to 1 row x ncolumns
# 		hs = hs + h * s

# 	return hs

# #########################################################################
# def read_sparse_h(timestep, config, getSparse=True, hdir=None):
# 	""" Read a sparse h slice file and return a sparse matrix.
# 	The h file can be either a numpy .npz file (created using hsplit.py)
# 	(or a .bin binary file (created with write_binsparse())
# 	"""

# 	if hdir is None:
# 		Hdir = config["workdir"] + "/" + config["hdir"]
# 	else:
# 		Hdir = hdir

# 	hfile = Hdir + "/H%04d.npz" % (timestep+1)

# 	if os.path.exists(hfile):
# 		a = numpy.load(hfile)
# 		if getSparse:
# 			try:
# 				h = csr_matrix((a['vals'], (a['rows'], a['cols'])), shape=a['shape'])
# 			except TypeError as err:
# 				sys.exit("ERROR: Can't create sparse matrix from H file %s. %s" % (hfile, err))
# 		else:
# 			h = numpy.zeros(a['shape'])
# 			h[a['rows'], a['cols']] = a['vals']
# 	else:
# 		hfile = Hdir + "/H%04d.bin" % (timestep+1)
# 		if os.path.exists(hfile):
# 			h = binread_sparse(hfile, getSparse)
# 		else:
# 			raise IOError("Did not find .npz or .bin file for H step %d" % timestep)

# 	return h


# #########################################################################
# # Mimic scipy.linalg  block_diag
# # This is available only for versions >=0.8, but we have 0.7.2,
# # so we define our own here
# #########################################################################
# def block_diag(*arrs):
# 	"""Create a block diagonal matrix from the provided arrays.

# 	Given the inputs `A`, `B` and `C`, the output will have these
# 	arrays arranged on the diagonal::

# 	[[A, 0, 0],
# 	 [0, B, 0],
# 	 [0, 0, C]]

# 	If all the input arrays are square, the output is known as a
# 	block diagonal matrix.

# 	Parameters
# 	----------
# 	A, B, C, ... : array-like, up to 2D
# 	Input arrays.  A 1D array or array-like sequence with length n is
# 	treated as a 2D array with shape (1,n).

# 	Returns
# 	-------
# 	D : ndarray
# 	Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
# 	same dtype as `A`.

# 	References
# 	----------
# 	.. [1] Wikipedia, "Block matrix",
# 	   http://en.wikipedia.org/wiki/Block_diagonal_matrix

# 	Examples
# 	--------
# 	>>> A = [[1, 0],
# 	...      [0, 1]]
# 	>>> B = [[3, 4, 5],
# 	...      [6, 7, 8]]
# 	>>> C = [[7]]
# 	>>> print(block_diag(A, B, C))
# 	[[1 0 0 0 0 0]
# 	[0 1 0 0 0 0]
# 	[0 0 3 4 5 0]
# 	[0 0 6 7 8 0]
# 	[0 0 0 0 0 7]]
# 	>>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
# 	array([[ 1.,  0.,  0.,  0.,  0.],
# 	   [ 0.,  2.,  3.,  0.,  0.],
# 	   [ 0.,  0.,  0.,  4.,  5.],
# 	   [ 0.,  0.,  0.,  6.,  7.]])

# 	"""
# 	if arrs == ():
# 		arrs = ([],)
# 	arrs = [numpy.atleast_2d(a) for a in arrs]

# 	bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
# 	if bad_args:
# 		raise ValueError("arguments in the following positions have dimension "
# 			    "greater than 2: %s" % bad_args)

# 	shapes = numpy.array([a.shape for a in arrs])
# 	out = numpy.zeros(numpy.sum(shapes, axis=0), dtype=arrs[0].dtype)

# 	r, c = 0, 0
# 	for i, (rr, cc) in enumerate(shapes):
# 		out[r:r + rr, c:c + cc] = arrs[i]
# 		r += rr
# 		c += cc
# 	return out

# #########################################################################
# def getSpriorNetcdf(filename, varname, landmask):
# 	""" Get sprior grid values from a netcdf file.
# 	Use landmask to extract land cell values.
# 	Assumes that the netcdf sprior grid is compatible with landmask, e.g. 70x120 1 degree cells.
# 	"""

# 	try:
# 		ds = netCDF4.Dataset(filename)
# 	except IOError as e:
# 		sys.exit("Error trying to read netcdf file %s." % filename)

# 	g = ds.variables[varname][:]
# 	sprior = g[:, landmask[0], landmask[1]]

# 	ds.close()

# 	return sprior


# ####################################################
# def extract_grid(grid, lats, lons, config):
# 	""" extract the inversion domain from the footprint domain
# 	grid - footprint grid [ntimesteps x nlatx x nlons]
# 	lats - the latitudes in the footprint
# 	lons - the longitudes in the footprint
# 	config - config file data

# 	Only works with 1 degree grids
# 	"""

# 	# get the minimum latitude and longitude
# 	# use math.floor in case the latitudes are centered on a grid cell,
# 	# such that the value has .5 in it, e.g. -169.5
# 	latmin = math.floor(lats[0])
# 	lonmin = math.floor(lons[0])

# 	# determine the offset from the footprint domain where the inversion domain is located
# 	# if the footprint domain and inversion domain are the same, bottom and left are 0,
# 	# top and right are size of domain
# 	by = config['south'] - latmin		# bottom
# 	ty = config['north'] - latmin		# top
# 	lx = config['west'] - lonmin		# left
# 	rx = config['east'] - lonmin		# right


# 	mygrid = grid[:, by:ty, lx:rx]

# 	return mygrid
