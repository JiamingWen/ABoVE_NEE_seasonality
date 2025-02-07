#create a shapefile of ABoVE based on lat/lon boundary, used for AppEEARs downloads
setwd('C:/Users/jwen/Research/ABoVE/above_mask/')
library(sp)
library(sf)

# library(rgdal) # not available
library(raster)

lon = -135; offset1 = 35
lat = 62.5; offset2 = 12.5
RegionName = 'ABoVE_latlon_boundary'

x_coor=c(lon-offset1,lon+offset1,lon+offset1,lon-offset1)
y_coor=c(lat+offset2,lat+offset2,lat-offset2,lat-offset2)
xym <- cbind(x_coor, y_coor)
p = Polygon(xym)
ps = Polygons(list(p),1)
sps = SpatialPolygons(list(ps))
proj4string(sps) = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")
plot(sps)
data_dummy = data.frame(f=99.9)
spdf = SpatialPolygonsDataFrame(sps,data_dummy)
dir.create('Shapefile')

# writeOGR(spdf, dsn = paste0(RegionName,"/Shapefile"), layer = RegionName, driver = "ESRI Shapefile")
shapefile(spdf, filename=paste0('Shapefile/',RegionName,'.shp'))
