# install.packages("BiocManager")
# require("BiocManager")
# BiocManager::install("rhdf5")

library(rhdf5)
library(stringr)
library(raster)

filename1 = 'C:/Users/jwen/Research/ABoVE/gfed/GFED4.1s_2016.hdf5'
h5ls(filename1)

generate_raster = function(array0){
  ras = raster(res=0.25)
  ras[] = as.vector(array0)
  return(ras)
}

month = 7
monthstr = str_pad(month, 2, pad='0')
C_emission = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/C"))) #unit g C m-2 month-1
DM_emission = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/DM"))) #unit kg DM m-2 month-1

C_AGRI_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/C_AGRI"))) #range 0-1
C_BORF_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/C_BORF"))) #range 0-1
C_DEFO_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/C_DEFO"))) #range 0-1
C_PEAT_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/C_PEAT"))) #range 0-1
C_SAVA_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/C_SAVA"))) #range 0-1
C_TEMF_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/C_TEMF"))) #range 0-1

DM_AGRI_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/DM_AGRI"))) #range 0-1
DM_BORF_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/DM_BORF"))) #range 0-1
DM_DEFO_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/DM_DEFO"))) #range 0-1
DM_PEAT_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/DM_PEAT"))) #range 0-1
DM_SAVA_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/DM_SAVA"))) #range 0-1
DM_TEMF_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/DM_TEMF"))) #range 0-1

emission_factor = read.table('C:/Users/jwen/Research/ABoVE/gfed/GFED4_Emission_Factors.txt', row.names = 1, col.names = c('C_species','AGRI', 'BORF', 'DEFO', 'PEAT', 'SAVA', 'TEMF'))

#check C emissions
#calculate C emission from C
ras1 = C_emission * C_TEMF_frac
summary(ras1[])
#calculate C emission from DM
ras2 = DM_emission * DM_TEMF_frac * emission_factor['C', 'TEMF']
summary(ras2[])
summary(ras1[]-ras2[])

ras1[ras1 == 0] = NA
plot(ras1, colNA='black')

ras2[ras2 == 0] = NA
plot(ras2, colNA='black')

dif = ras1[] - ras2[]
dif[dif>0 & !is.na(dif)] / ras2[dif>0 & !is.na(dif)]

#check CO2 emissions
CO2_ras1 = raster(res=0.25); CO2_ras1[] = 0
CO2_ras2 = raster(res=0.25); CO2_ras2[] = 0

for (LC in c('AGRI', 'BORF', 'DEFO', 'PEAT', 'SAVA', 'TEMF')){
  C_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/C_", LC))) #range 0-1
  DM_frac = generate_raster(h5read(file = filename1, name = paste0("emissions/", monthstr, "/partitioning/DM_", LC))) #range 0-1
  
  CO2_ras1 = CO2_ras1 + C_emission * C_frac / emission_factor['C', LC] * emission_factor['CO2', LC]
  CO2_ras2 = CO2_ras2 + DM_emission * DM_frac * emission_factor['CO2', LC]
}

summary(CO2_ras1[])
summary(CO2_ras2[])
summary(CO2_ras1[]-CO2_ras2[])

CO2_ras1[CO2_ras1 == 0] = NA
plot(CO2_ras1, colNA='black')

CO2_ras2[CO2_ras2 == 0] = NA
plot(CO2_ras2, colNA='black')


dif = CO2_ras1[] - CO2_ras2[]
dif[dif>0.01 & !is.na(dif)] / CO2_ras2[dif>0.01 & !is.na(dif)]


##################################################
library(ncdf4)
ncin = nc_open('C:/Users/jwen/Research/ABoVE/gfed/GFED4.1s-quarter-degree-2016.nc')
print(ncin)
CO2_emission_array = ncvar_get(ncin, 'CO2_emission')
ras = raster(res=0.25)
ras[] = as.vector(CO2_emission_array[,,7])
ras = flip(ras, 2)
ras[ras == 0] = NA
plot(ras, colNA='black')
summary(ras[])

dif = ras - CO2_ras2
plot(dif, colNA='black')
summary(dif[])
summary(dif[]/ras[])

#regrid to half degree
ras = raster('C:/Users/jwen/Research/ABoVE/gfed/GFED4.1s-half-degree-2016.nc',band=7)
ras[ras == 0] = NA
plot(ras, colNA='black')

CO2_ras2_agg = aggregate(CO2_ras2, fact=2, fun=mean) #need to re-run CO2_ras2 and do not set zero values to nan
ras[ras == 0] = NA
plot(ras, colNA='black')

dif = ras - CO2_ras2_agg
plot(dif, colNA='black')
summary(dif[])
summary(dif[]/ras[])
