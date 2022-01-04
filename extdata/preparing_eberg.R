## preparing Eberg extra layers

library(plotKML)
data("eberg_grid25")
gridded(eberg_grid25) <- ~x+y
proj4string(eberg_grid25) <- CRS("+init=epsg:31467")
r <- raster(eberg_grid25)
data(eberg_zones)
library(sf)
library(fasterize)
eberg_zones_sf <- as(eberg_zones, "sf")
eberg_zones_r <- fasterize(eberg_zones_sf, r, field="ZONES")
levels(eberg_zones$ZONES)
#> [1] "Clay_and_loess"  "Clayey_derivats" "Sandy_material"  "Silt_and_sand"
eberg_zones_r = as(eberg_zones_r, "SpatialGridDataFrame")
eberg_zones_r$ZONES <- as.factor(eberg_zones_r$layer)
levels(eberg_zones_r$ZONES) <- levels(eberg_zones$ZONES)
## derive TWI etc in SAGA GIS
library(rgdal)
writeGDAL(eberg_grid25["rmspe.cly.eml"], "/data/tmp/eberg_rmspe.cly.eml.sdat", drivername = "SAGA")
writeGDAL(eberg_grid25["pred.cly.eml"], "/data/tmp/eberg_pred.cly.eml.sdat", drivername = "SAGA")
unlink("/data/tmp/eberg_points.gpkg")
writeOGR(eberg.sp, "/data/tmp/eberg_points.gpkg", "", "GPKG")
grd = raster::stack(list.files("/data/tmp", glob2rx("eberg_*.sdat"), full.names=TRUE))
grd = as(grd, "SpatialGridDataFrame")
## save to a RDS file:
grd$PMTZONES = eberg_zones_r$ZONES
saveRDS(grd, "./extdata/eberg_dtm_25m.rds")

## Edgeroi
## DON'T run
r100m = raster(edgeroi.grids100[1])
?edgeroi
rep = "https://raw.github.com/Envirometrix/PredictiveSoilMapping/master/extdata/"
x = tempfile(fileext = ".rda")
con <- download.file(paste0(rep, "edgeroi.grids.rda"), x, method="wget")
load(x)
str(edgeroi.grids)
gridded(edgeroi.grids) <- ~x+y
proj4string(edgeroi.grids) <- CRS("+init=epsg:28355")
edgeroi.grids.r = raster::resample(stack(edgeroi.grids[-3]), r100m, method='bilinear')
edgeroi.grids.100m = cbind(edgeroi.grids100, as(edgeroi.grids.r, "SpatialGridDataFrame"))
str(edgeroi.grids.100m@data)
saveRDS(edgeroi.grids.100m, "./extdata/edgeroi.grids.100m.rds")
