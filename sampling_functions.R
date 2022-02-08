## auxiliary functions
## tom.hengl@opengeohub.org

hor2xyd <- function(x, U="UHDICM", L="LHDICM", treshold.T=15){
  x$DEPTH <- x[,U] + (x[,L] - x[,U])/2
  x$THICK <- x[,L] - x[,U]
  sel <- x$THICK < treshold.T
  ## begin and end of the horizon:
  x1 <- x[!sel,]; x1$DEPTH = x1[,L]
  x2 <- x[!sel,]; x2$DEPTH = x1[,U]
  y <- do.call(rbind, list(x, x1, x2))
  return(y)
}

comp.var = function(x, r1, r2, v1, v2){
  r =  rowSums( x[,c(r1, r2)] * 1/(x[,c(v1, v2)]^2) ) / rowSums( 1/(x[,c(v1, v2)]^2) )
  ## https://stackoverflow.com/questions/13593196/standard-deviation-of-combined-data
  v = sqrt( rowMeans(x[,c(r1, r2)]^2 + x[,c(v1, v2)]^2) - rowMeans( x[,c(r1, r2)])^2 )
  return(data.frame(response=r, stdev=v))
}

pfun <- function(x,y, ...){
  panel.hexbinplot(x,y, ...)
  panel.abline(0,1,lty=1,lw=2,col="black")
}

plot_hexbin <- function(varn, breaks, main, meas, pred, colorcut=c(0,0.01,0.03,0.07,0.15,0.25,0.5,0.75,1), pal = openair::openColours("increment", 18)[-18], in.file, log.plot=FALSE, out.file){ ## pal=R_pal[["bpy_colors"]][1:18]
  require("hexbin"); require("plotKML"); require("latticeExtra"); require("openair")
  if(missing(out.file)){ out.file = paste0("./img/plot_CV_", varn, ".png") }
  if(!file.exists(out.file)){
    if(missing(pred)){
      m <- readRDS.gz(in.file)
      #pred <- t.m$learner.model$super.model$learner.model$fitted.values
      pred <- m$predictions
    }
    if(missing(meas)){
      meas <- t.m$learner.model$super.model$learner.model$model[,1]
    }
    if(log.plot==TRUE){
      R.squared = yardstick::ccc(data.frame(pred, meas), truth="meas", estimate="pred")
      #pred <- 10^pred-1
      pred = expm1(pred)
      #meas <- 10^meas-1
      meas = expm1(meas)
      d.meas <- min(meas, na.rm=TRUE)
    } else {
      d.meas <- min(meas, na.rm=TRUE)
      R.squared = yardstick::ccc(data.frame(pred, meas), truth="meas", estimate="pred")
    }
    main.txt = paste0(main, "  (CCC: ", signif(R.squared$.estimate, 3), ")")
    png(file = out.file, res = 150, width=850, height=850, type="cairo")
    if(log.plot==TRUE){
      pred <- pred+ifelse(d.meas==0, 1, d.meas)
      meas <- meas+ifelse(d.meas==0, 1, d.meas)
      lim <- range(breaks)+ifelse(d.meas==0, 1, d.meas)
      meas <- ifelse(meas<lim[1], lim[1], ifelse(meas>lim[2], lim[2], meas))
      plt <- hexbinplot(meas~pred, colramp=colorRampPalette(pal), main=main.txt, ylab="measured", xlab="predicted", type="g", lwd=1, lcex=8, inner=.2, cex.labels=.8, scales=list(x = list(log = 2, equispaced.log = FALSE), y = list(log = 2, equispaced.log = FALSE)), asp=1, xbins=50, ybins=50, xlim=lim, ylim=lim, panel=pfun, colorcut=colorcut)
    } else {
      lim <- range(breaks)
      meas <- ifelse(meas<lim[1], lim[1], ifelse(meas>lim[2], lim[2], meas))
      plt <- hexbinplot(meas~pred, colramp=colorRampPalette(pal), main=main.txt, ylab="measured", xlab="predicted", type="g", lwd=1, lcex=8, inner=.2, cex.labels=.8, xlim=lim, ylim=lim, asp=1, xbins=50, ybins=50, panel=pfun, colorcut=colorcut)
    }
    print(plt)
    dev.off()
  }
}

## mapping wrapper functions


plot.map <- function(df, title="", fill=SAGA_pal[[1]], points, pcex=4, maptype="", grid=3, points2=NULL){
  
  df<- df_prep(df)
  
  if("LONGDA94" %in% names(as.data.frame(points))){
    points <- as.data.frame(points,xy=T) %>% rename(x=LONGDA94,y=LATGDA94)
  }
  names(points)<- c("x","y")
  names(points) <- tolower(names(points))
  
  map <- ggplot()
  
  if (maptype == "grey") {
    map <- map + geom_raster(data = df , aes(x = x, y = y), fill = "grey")
  } else{
    map <- map + geom_raster(data = df , aes(x = x, y = y, fill = value))
  }
  if(!is.null(points2)){
    points2<- as.data.frame(points2, xy=TRUE)
    names(points2) <- c("x","y")
    map <- map+geom_point(data = points2, aes(x=x,y=y), shape=19, size=2, colour="blue")
  }
  
  map + geom_point(data = points, aes(x = x, y = y), shape = "+",size = pcex) +
    scale_fill_gradientn(colours  = fill, name = "") +
    coord_sf() +
    ggtitle(title) +
    theme_minimal() +
    xlab("Easting") +
    ylab("Northing") +
    scale_x_continuous(breaks = pretty(df$x, n = grid)) +
    scale_y_continuous(breaks = pretty(df$y, n = grid))
  
  }



compare.map <- function(df, title = "",  fill=SAGA_pal[[1]], points=NULL,grid=3, pcex=4) {
  
  df<- df %>% as.data.frame(xy = TRUE) %>% gather(key = layer, value = value,-x,-y)
  
  if("LONGDA94" %in% names(as.data.frame(points))){
    points <- as.data.frame(points,xy=T) %>% rename(x=LONGDA94,y=LATGDA94)
  }
  map <- ggplot() +
    geom_raster(data = df , aes(x = x, y = y, fill = value)) +
    scale_fill_gradientn(colours = fill, name = "") +
    facet_wrap(layer ~ .) +
    coord_sf() +
    ggtitle(title) +
    theme_minimal() +
    xlab("Easting") + 
    ylab("Northing")+
    theme(panel.spacing = unit(2, "lines"))+
    scale_x_continuous(breaks = pretty(df$x, n = grid)) +
    scale_y_continuous(breaks = pretty(df$y, n = grid))
  
  if(!is.null(points)){
    names(points) <- tolower(names(points))
    map+geom_point(data = points, aes(x=x,y=y), shape="+", size=pcex)
  }else{
    map
  }
}


df_prep <- function(df) {
  if (length(names(df)) == 1 & names(df) =="PC1") {
    df <- df %>% as.data.frame(xy = TRUE)
    names(df) <- c("x","y","value")
  } 
   else if("gid" %in% names(df)){
     df <- as.data.frame(df, xy = TRUE)
     names(df) <- c("value","x","y")
   }else if (length(names(df)) == 1 & names(df) == "pr.VW") {
     df <- df %>% as.data.frame(xy = TRUE)
     names(df) <- c("x","y","value")
   } 
  else if("s1" %in% names(as.data.frame(df))){
    df <- as.data.frame(df, xy = TRUE)
    names(df) <- c("value","x","y")
  }
  else{
    df <- df %>% as.data.frame(xy = TRUE) %>% gather(key = layer, value = value,-x,-y)
    
  }
  df <- na.omit(df)
  return(df)
}




plot_hex <- function(df, pts,title=""){
  reds = RColorBrewer::brewer.pal(9, "YlOrRd")[-1]
  df = as.data.frame(df)
  pts= as.data.frame(pts)
  
  ggplot(df, aes(x = PC1, y=PC2)) +
    stat_binhex(bins = 60)+
    #scale_fill_distiller(palette = "YlOrRd",direction = 1)+
    guides(fill = guide_coloursteps(show.limits = TRUE, title="Counts",barheight= grid::unit(6,"cm") ))+
    theme_minimal()+
    coord_equal()+
    geom_point(data=pts, aes(x = PC1, y=PC2), shape="+", size=3)+
    ggtitle(title)+
    binned_scale(aesthetics = "fill", scale_name = "custom",
                 palette = ggplot2:::binned_pal(scales::manual_pal(values = reds)),
                 guide = "bins", breaks = seq(0,800,100), show.limits = TRUE)
  
}

