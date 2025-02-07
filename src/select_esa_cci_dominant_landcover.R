#plot ESA-CCI land cover map

library(raster)
library(ncdf4)
setwd('C:/Users/jwen/Research/ABoVE/esa_cci_landcover/')

ncin = nc_open('esa-cci-land-cover-2017-half-degree-combined-classes.nc')
print(ncin)
lccs_class_frac_combined = ncvar_get(ncin, 'lccs_class_frac_combined')

# determine the dominant land cover
determine_dominant_landcover_esacci = function(vec){
  id = 0:15
  id[which.max(vec)]
}

dominant_array = apply(lccs_class_frac_combined, c(1,2), determine_dominant_landcover_esacci)

ras = raster(res = 0.5)
ras[] = as.vector(dominant_array)
ras = flip(ras, 2)
plot(ras)
writeRaster(ras, filename = 'esa-cci-dominant-landcover-2017.nc', overwrite=T)


##################################################################
lcnew = raster('esa-cci-dominant-landcover-2017.nc')

lc_longname=c('Cultivated',
              'Generic forests',
              'Evergreen broadleaf forests',
              'Deciduous broadleaf forests',
              'Evergreen needleleaf forests',
              'Deciduous needleleaf forests',
              'Shrublands',
              'Grasslands',
              'Lichens and mosses',
              'Sparsely vegetated',
              'Wetlands',
              'Developed',
              'Barren',
              'Water',
              'Snow and ice')

#plot
library(lattice)
library(rasterVis)
lcnew2=as.factor(lcnew)
rat <- levels(lcnew2)[[1]]
rat[["landcover"]] <- lc_longname
levels(lcnew2) <- rat
palette = c('#c24f44', '#009900', '#086a10', '#78d203', '#05450a', '#54a708','#c6b044','#b6ff05', 
            '#ffafdc', '#4d3131', '#27ff87', '#a5a5a5', '#f9ffa4', '#1c0dff', '#69fff8')

levelplot(lcnew2,margin=F,colorkey=T,at=1:nrow(rat),col.regions=palette,xlab="",ylab="",scales=list(x=list(at=seq(-180,180,30)),y=list(at=seq(-180,180,30))))

#crop the ABoVE region
above_mask = raster('../above_mask/above_mask.nc')
lcnew2[above_mask[] != 0] = NA
lcnew2_cp = crop(lcnew2, raster(res=0.5, xmn=-170, xmx=-100, ymn=50, ymx=75))
levelplot(lcnew2_cp,margin=F,colorkey=T,at=1:nrow(rat),col.regions=palette,main='ESA-CCI LC',xlab="",ylab="",scales=list(x=list(at=seq(-180,-100,20)),y=list(at=seq(50,80,5))))
