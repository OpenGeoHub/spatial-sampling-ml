# Generating spatial sampling

::: {.rmdnote}
You are reading the work-in-progress Spatial Sampling and Resampling for Machine Learning. This chapter is currently currently draft version, a peer-review publication is pending. You can find the polished first edition at <https://opengeohub.github.io/spatial-sampling-ml/>.
:::



## Spatial sampling

Sampling in statistics is done for the purpose of estimating population
parameters and/or for testing of experiments. If Observations and
Measurements (O&M) are collected in space i.e. as geographical variables
this is referred to as **spatial sampling** and is often materialized as
**a point map** with points representing locations of planned or
implemented O&M. Preparing a spatial sampling plan is a type of **design
of experiment** and hence it is important to do it right to avoid any
potential bias.

Spatial sampling or producing and implementing sampling designs are
common in various fields including physical geography, soil science,
geology, vegetation science, ecology and similar. Imagine an area that
potentially has problems with soil pollution by heavy metals. If we
collect enough samples, we can overlay points vs covariate layers, then
train spatial interpolation / spatial prediction models and produce
predictions of the target variable. For example, to map soil pollution
by heavy metals or soil organic carbon stock, we can collect soil
samples on e.g. a few hundred predefined locations, then take the
samples to the lab, measure individual values and then interpolate them
to produce a map of concentrations. This is one of the most common
methods of interest of **geostatistics** where e.g. various [kriging methods](http://www.leg.ufpr.br/geor/)
are used to produce predictions of the target variable (see e.g.
@Bivand2013Springer).

There are many sampling design algorithms that can be used to spatial
sampling locations. In principle, all spatial sampling approaches can be
grouped based on the following four aspects:

1.  *How objective is it?* Here two groups exist:

    a.  Objective sampling designs which are either [***probability
        sampling***](https://towardsdatascience.com/an-introduction-to-probability-sampling-methods-7a936e486b5)
        or some experimental designs from spatial statistics;

    b.  Subjective or **convenience sampling** which means that the
        inclusion probabilities are unknown and are often based on
        convenience e.g. distance to roads / accessibility;

2.  *How much identically distributed is it*? Here at least three groups
    exist:

    a.  **[Independent Identically Distributed (IID)](https://xzhu0027.gitbook.io/blog/ml-system/sys-ml-index/learning-from-non-iid-data)**
        sampling designs,

    b.  Clustered sampling i.e. non-equal probability sampling,

    c.  Censored sampling,

3.  *Is it based on geographical or feature space?* Here at least three
    groups exist:

    a.  Geographical sampling i.e. taking into account only geographical dimensions + time;

    b.  **Feature-space sampling** i.e. taking into account only distribution of points in feature space;

    c.  Hybrid sampling i.e. taking both feature and geographical space into account;

4.  *How optimized is it?* Here at least two groups exist:

    a.  **Optimized sampling** so that the target optimization criteria 
        reaches minimum / maximum i.e. it can be proven as being optimized,

    b.  *Unoptimized sampling*, when either optimization criteria can not be tested or is unknown,

Doing sampling using objective sampling designs is important as it
allows us to test hypotheses and produce **unbiased estimation** of
population parameters or similar. Many spatial statisticians argue that
only previously prepared, strictly followed randomized probability
sampling can be used to provide an unbiased estimate of the accuracy of
the spatial predictions [@Brus2021sampling]. In the case
of probability sampling, calculation of population parameters is derived
mathematically i.e. that estimation process is unbiased and independent
of the spatial properties of the target variable (e.g. spatial
dependence structure and/or statistical distribution). For example, if
we generate sampling locations using **Simple Random Sampling (SRS)**,
this sampling design has the following properties:

1.  It is an IID with each spatial location with exactly the same
    **inclusion probability**,
2.  It is symmetrical in geographical space meaning that about the same
    number of points can be found in each quadrant of the study area,
3.  It can be used to derive population parameters (e.g. mean) and these
    measures are per definition unbiased,
4.  Any random subset of the SRS is also a SRS,

SRS is in principle both objective sampling and IID sampling and can be
easily generated provided some polygon map representing the study area.
Two other somewhat more complex sampling algorithms with similar
properties as SRS are for example different versions of tessellated
sampling. The **generalized random tessellation stratified (GRTS)
design** was for example used in the USA to produce sampling locations
for the purpose of [geochemical mapping](https://pubs.usgs.gov/ds/801/); 
a **multi-stage stratified random sampling** design was used to produce [LUCAS soil monitoring
network of points](https://esdac.jrc.ec.europa.eu/projects/lucas).

Large point datasets representing observations and/or measurements 
*in-situ* can be used to generate maps by fitting regression and 
classification models using e.g. Machine Learning algorithms, then 
applying those models to predict values at all pixels. This is referred 
to as [***Predictive Mapping***](https://soilmapper.org). In reality, 
many point datasets we use in Machine Learning for predictive mapping do 
not have ideal properties i.e. are neither IID nor are probabilities of 
inclusion known. Many are in fact purposive, **[convenience sampling](https://methods.sagepub.com/reference/encyclopedia-of-survey-research-methods/n105.xml)**
and hence potentially over-represent some geographic features, are
potentially censored and can lead to significant bias in estimation.

## Response surface designs

If the objective of modeling is to build regression models (correlating
the target variable with a number of spatial layers representing e.g.
soil forming factors), then we are looking at the problem in statistics
known as the **[response-surface experimental designs](https://en.wikipedia.org/wiki/Response_surface_methodology)**.
Consider the following case of one target variable ($Y$) and one
covariate variable ($X$). Assuming that the two are correlated linearly
(i.e. $Y = b_0 + b_1 \cdot X$), one can easily prove that the optimal
experimental design is to (a) determine min and max of $X$, the put half
of the point at $X_{min}$ and the other half at $X_{max}$ [@Hengl2004AJSR]. This design is called the
**[D1 optimal design](https://en.wikipedia.org/wiki/Optimal_design#D-optimality)** and
indeed it looks relatively simple to implement. The problem is that it
is the optimal design ONLY if the relationship between $Y$ and $X$ is
perfectly linear. If the relationship is maybe close to quadratic than
the D1 design is much worse than the D2 design.

<div class="figure" style="text-align: center">
<img src="./img/Fig_D1_design_scheme.jpg" alt="Example of D1 design: (left) D1 design in 2D feature space, (right) D1 design is optimal only for linear model, if the model is curvilinear, it is in fact the worse design than simple random sampling." width="90%" />
<p class="caption">(\#fig:scheme-d1)Example of D1 design: (left) D1 design in 2D feature space, (right) D1 design is optimal only for linear model, if the model is curvilinear, it is in fact the worse design than simple random sampling.</p>
</div>

In practice we may not know what is the nature of the relationship
between $Y$ and $X$, i.e. we do not wish to take a risk and produce
biased estimation. Hence we can assume that it could be a curvilinear
relationship and so we need to sample uniformly in the feature space.

## Spatial sampling algorithms of interest

This chapter reviews some common approaches for preparing point samples for a 
study area that you are visiting for the first time and/or no previous samples or 
models are available. We focus on the following spatial sampling methods: 

- Subjective or convenience sampling,
- Simple Random Sampling (**SRS**) [@Bivand2013Springer;@Brus2021sampling],
- Latin Hypercube Sampling (**LHS**) and its variants e.g. Conditioned LHS [@minasny2006conditioned;@Malone2019PeerJ],
- Feature Space Coverage Sampling (**FSCS**) [@Goerg2013] and fuzzy k-means clustering [@hastie2009elements],
- 2nd round sampling [@stumpf2017uncertainty],

Our interest in this tutorials is in producing predictions (maps) of the target 
variable by employing regression / correlation between the target variable and 
multitude of features (raster layers), and where various Machine Learning techniques 
are used for training and prediction.

Once we collect enough training points in an area we can overlay points and GIS 
layers to produce a **regression matrix** or **classification matrix**, and 
which can then be used to generate spatial predictions i.e. produce maps. As a 
state-of-the-art we use the mlr framework for Ensemble Machine Learning as the key 
Machine Learning framework for predictive mapping. For an introduction to Ensemble 
Machine Learning for Predictive Mapping please refer to [this tutorial](https://gitlab.com/openlandmap/spatial-predictions-using-eml).

## Ebergotzen dataset

To test various sampling and mapping algorithms, we can use the Ebergotzen 
dataset available also via the **plotKML** package [@hengl2015plotkml]:


```r
set.seed(100)
library(plotKML)
library(sp)
library(viridis)
#> Loading required package: viridisLite
library(raster)
library(ggplot2)
data("eberg_grid25")
gridded(eberg_grid25) <- ~x+y
proj4string(eberg_grid25) <- CRS("+init=epsg:31467")
```

This dataset is described in detail in @Bohner2008Hamburg. It is a soil survey 
dataset with ground observations and measurements of soil properties including soil 
types. The study area is a perfect square 10×10 km in size.

We have previously derived number of additional DEM parameters directly using SAGA GIS 
[@Conrad2015] and which we can add to the list of covariates:


```r
eberg_grid25 = cbind(eberg_grid25, readRDS("./extdata/eberg_dtm_25m.rds"))
names(eberg_grid25)
#>  [1] "DEMTOPx"        "HBTSOLx"        "TWITOPx"        "NVILANx"       
#>  [5] "eberg_dscurv"   "eberg_hshade"   "eberg_lsfactor" "eberg_pcurv"   
#>  [9] "eberg_slope"    "eberg_stwi"     "eberg_twi"      "eberg_vdepth"  
#> [13] "PMTZONES"
```

so a total of 11 layers at 25-m spatial resolution, from which two layers 
(`HBTSOLx` and `PMTZONES`) are factors representing soil units and parent material 
units. Next, for further analysis, and to reduce data overlap, we can convert all 
primary variables to (numeric) principal components using:


```r
eberg_spc = landmap::spc(eberg_grid25[-c(2,3)])
#> Converting PMTZONES to indicators...
#> Converting covariates to principal components...
```

which gives the a total of 14 PCs. The patterns in the PC components reflect 
a complex combination of terrain variables, lithological discontinuities (`PMTZONES`) 
and surface vegetation (`NVILANx`):


```r
spplot(eberg_spc@predicted[1:3], col.regions=SAGA_pal[[1]])
```

<div class="figure" style="text-align: center">
<img src="./img/Fig_PCA_components_Ebergotzen.png" alt="Principal Components derived using Ebergotzen dataset." width="100%" />
<p class="caption">(\#fig:plot-spc)Principal Components derived using Ebergotzen dataset.</p>
</div>

## Simple Random Sampling

To generate a SRS with e.g. 100 points we can use the **sp** package `spsample` method:


```r
rnd <- spsample(eberg_grid25[1], type="random", n=100)
```


```r
plot(raster(eberg_spc@predicted[1]), col=SAGA_pal[[1]])
points(rnd, pch="+")
```

<div class="figure" style="text-align: center">
<img src="sampling_files/figure-html/eberg-srs-1.png" alt="An example of a Simple Random Sample (SRS)." width="80%" />
<p class="caption">(\#fig:eberg-srs)An example of a Simple Random Sample (SRS).</p>
</div>

This sample is generated purely based on the spatial domain, the feature space 
is completely ignored / not taken into account, hence we can check how well do 
these points represent the feature space using a density plot:


```r
ov.rnd = sp::over(rnd, eberg_spc@predicted[,1:2])
library(hexbin)
library(grid)
reds = colorRampPalette(RColorBrewer::brewer.pal(9, "YlOrRd")[-1])
hb <- hexbin(eberg_spc@predicted@data[,1:2], xbins=60)
p <- plot(hb, colramp = reds, main='PCA Ebergotzen SRS')
pushHexport(p$plot.vp)
grid.points(ov.rnd[,1], ov.rnd[,2], pch="+")
```

<div class="figure" style="text-align: center">
<img src="sampling_files/figure-html/eberg-fs-1.png" alt="Distribution of the SRS points from the previous example in the feature space." width="80%" />
<p class="caption">(\#fig:eberg-fs)Distribution of the SRS points from the previous example in the feature space.</p>
</div>

Visually, we do not directly see from Fig. \@ref(fig:eberg-fs) that the generated SRS 
maybe misses some important feature space, however if we zoom in, then we can notice that some 
parts of feature space (in this specific randomization) with high density are somewhat 
under-represented. Imagine if we reduce number of sampling points then we run 
even higher risk of missing some areas in the feature space by using SRS.

Next, we are interested in evaluating the occurrence probability of the SRS points 
based on the PCA components. To derive the occurrence probability we can use 
the **maxlike** package method [@Royle2012]:


```r
fm.cov <- stats::as.formula(paste("~", paste(names(eberg_spc@predicted[1:4]), collapse="+")))
ml <- maxlike::maxlike(formula=fm.cov, rasters=raster::stack(eberg_spc@predicted[1:4]), 
                       points=rnd@coords, method="BFGS", removeDuplicates=TRUE, savedata=TRUE)
ml.prob <- predict(ml)
```


```r
plot(ml.prob)
points(rnd@coords, pch="+")
```

<div class="figure" style="text-align: center">
<img src="sampling_files/figure-html/eberg-maxlike-1.png" alt="Occurrence probability for SRS derived using the maxlike package." width="80%" />
<p class="caption">(\#fig:eberg-maxlike)Occurrence probability for SRS derived using the maxlike package.</p>
</div>

Note: for the sake of reducing the computing intensity we focus on the first four PCs. 
In practice, feature space analysis can be quite computational and we recommend using 
parallelized versions within an High Performance Computing environments to run such analysis.

Fig. \@ref(fig:eberg-maxlike) shows that, by accident, some parts of the feature 
space might be somewhat under-represented (areas with low probability of occurrence on the map). 
Note however that occurrence probability for this dataset is overall _very low_ (<0.05), 
indicating that distribution of points is not much correlated with the features. 
This specific SRS is thus probably satisfactory also for feature space analysis: 
the SRS points do not seem to group (which would in this case be by accident) and 
hence maxlike gives very low probability of occurrence.

We can repeat SRS many times and then see if the clustering of points gets more 
problematic, but as you can image, in practice for large number of samples it is 
a good chance that all features would get well represented also in the feature 
space even though we did not include feature space variables in the production of the 
sampling plan.

We can now also load the actual points collected for the Ebergotzen case study:


```r
data(eberg)
eberg.xy <- eberg[,c("X","Y")]
coordinates(eberg.xy) <- ~X+Y
proj4string(eberg.xy) <- CRS("+init=epsg:31467")
ov.xy = sp::over(eberg.xy, eberg_grid25[1])
eberg.xy = eberg.xy[!is.na(ov.xy$DEMTOPx),]
sel <- sample.int(length(eberg.xy), 100)
eberg.smp = eberg.xy[sel,]
```

To quickly estimate spread of points in geographical and feature spaces, we can 
also use the function `spsample.prob` that calls both the kernel density function 
from the **[spatstat](https://rdrr.io/cran/spatstat.core/man/density.ppp.html)** package, and derives the probability of occurrence using the 
**[maxlike](https://rdrr.io/github/rbchan/maxlike/man/maxlike-package.html)** package: 


```r
iprob <- landmap::spsample.prob(eberg.smp, eberg_spc@predicted[1:4])
#> Deriving kernel density map using sigma 1010 ...
#> Deriving inclusion probabilities using MaxLike analysis...
```

In this specific case, the actual sampling points are much more clustered, so if we plot 
the two occurrence probability maps derived using maxlike next to each other we get:


```r
op <- par(mfrow=c(1,2))
plot(raster(iprob$maxlike), zlim=c(0,1))
points(eberg.smp@coords, pch="+")
plot(ml.prob, zlim=c(0,1))
points(rnd@coords, pch="+")
par(op)
```

<div class="figure" style="text-align: center">
<img src="sampling_files/figure-html/eberg-maxlike2-1.png" alt="Comparison occurrence probability for actual and SRS samples derived using the maxlike package." width="100%" />
<p class="caption">(\#fig:eberg-maxlike2)Comparison occurrence probability for actual and SRS samples derived using the maxlike package.</p>
</div>

The map on the left clearly indicates that most of the sampling points are 
basically preferentially located in the plain area, while the hillands are 
systematically under-sampled. This we can also cross-check by reading the 
description of the dataset in @Bohner2008Hamburg:

- the Ebergotzen soil survey points focus on agricultural land only,  
- no objective sampling design has been used, hence some points are clustered,  

This is also clearly visible from the _feature map_ plot where one part of the feature 
space seem to be completely omitted from sampling:


```r
ov2.rnd = sp::over(eberg.smp, eberg_spc@predicted[,1:2])
p <- plot(hb, colramp = reds, main='PCA Ebergotzen actual')
pushHexport(p$plot.vp)
grid.points(ov2.rnd[,1], ov2.rnd[,2], pch="+")
```

<div class="figure" style="text-align: center">
<img src="sampling_files/figure-html/eberg-fs2-1.png" alt="Distribution of the actual survey points from the previous example displayed in the feature space." width="80%" />
<p class="caption">(\#fig:eberg-fs2)Distribution of the actual survey points from the previous example displayed in the feature space.</p>
</div>

## Latin Hypercube Sampling

In the previous example we have shown how to implement SRS and then also evaluate it 
against feature layers. Often SRS represent very well feature space so it can be 
directly used for Machine Learning and with a guarantee of not making too much bias  
in predictions. To avoid, however, risk of missing out some parts of the feature space, 
and also to try to optimize allocation of points, we can generate a sample using the 
**[Latin Hypercube Sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling)** (LHS) method. In a nutshell, LHS methods are based on 
dividing the **Cumulative Density Function** (CDF) into _n_ equal partitions, and 
then choosing a random data point in each partition, consequently:

- CDF of LHS samples matches the CDF of population (hence unbiased representation),  
- Extrapolation in the feature space should be minimized,  

Latin Hypercube and sampling optimization using LHS is explained in detail in @minasny2006conditioned 
and @shields2016generalization. For @brus2015balanced, LHS is just a special case 
of **[balanced sampling](http://www.antongrafstrom.se/balancedsampling/)** i.e. sampling based on allocation in feature space. 
The [lhs package](https://github.com/bertcarnell/lhs) also contains numerous examples of 
how to implement LHS for (non-spatial) data.

Here we use an implementation of the LHS available in the **clhs** package [@Roudier2011]:


```r
library(clhs)
rnd.lhs = clhs::clhs(eberg_spc@predicted[1:4], size=100, iter=100, progress=FALSE)
```

This actually implements the so-called _“Conditional LHS"_ [@minasny2006conditioned], 
and can get quite computational for large stack of rasters, hence we manually limit the 
number of iterations to 100.

We can plot the LHS sampling plan:


```r
plot(raster(eberg_spc@predicted[1]), col=SAGA_pal[[1]])
points(rnd.lhs@coords, pch="+")
```

<div class="figure" style="text-align: center">
<img src="sampling_files/figure-html/eberg-clhs-1.png" alt="An example of a Latin Hypercube Sample (LHS)." width="80%" />
<p class="caption">(\#fig:eberg-clhs)An example of a Latin Hypercube Sample (LHS).</p>
</div>


```r
p <- plot(hb, colramp = reds, main='PCA Ebergotzen LHS')
pushHexport(p$plot.vp)
grid.points(rnd.lhs$PC1, rnd.lhs$PC2, pch="+")
```

<div class="figure" style="text-align: center">
<img src="sampling_files/figure-html/eberg-lhs2-1.png" alt="Distribution of the LHS points from the previous example displayed in the feature space." width="80%" />
<p class="caption">(\#fig:eberg-lhs2)Distribution of the LHS points from the previous example displayed in the feature space.</p>
</div>

Although in principle we might not see any difference in the point pattern between 
SRS and LHS, the feature space plot clearly shows that LHS covers systematically 
feature space map, i.e. we would have a relatively low risk of missing out some 
important features as compared to Fig. \@ref(fig:eberg-fs2). 

Thus the main advantages of the LHS are:

- it ensures that feature space is represented systematically i.e. it is optimized 
  for Machine Learning using the specific feature layers;  
- it is an **[Independent Identically Distributed (IID)](https://xzhu0027.gitbook.io/blog/ml-system/sys-ml-index/learning-from-non-iid-data)** sampling design;  
- thanks to the **clhs** package, also the survey costs raster layer can be integrated to still 
  keep systematic spread, but reduce survey costs as much as possible [@roudier2012conditioned];  

## Feature Space Coverage Sampling

The **Feature Space Coverage Sampling** (FSCS) is described in detail in @BRUS2019464.
In a nutshell, FSCS aims at optimal coverage of the feature space which is achieved 
by minimizing the average distance of the population units (raster cells) to the 
nearest sampling units in the feature space represented by the raster layers [@ma2020comparison].

To produce FSCS point sample we can use function `kmeanspp` of package **LICORS** [@Goerg2013].
First we partition the feature space cube to e.g. 100 clusters. We then select raster cells 
with the shortest scaled Euclidean distance in covariate-space to the centres of 
the clusters as the sampling units:


```r
library(LICORS)
library(fields)
fscs.clust <- kmeanspp(eberg_spc@predicted@data[,1:4], k=100, iter.max=100)
D <- fields::rdist(x1 = fscs.clust$centers, x2 = eberg_spc@predicted@data[,1:4])
units <- apply(D, MARGIN = 1, FUN = which.min)
rnd.fscs <- eberg_spc@predicted@coords[units,]
```

Note: the k-means++ algorithm is of most interest for small sample sizes: _“for large 
sample sizes the extra time needed for computing the initial centres can become 
substantial and may not outweigh the larger number of starts that can be afforded 
with the usual k-means algorithm for the same computing time”_ [@Brus2021sampling].
The `kmeanspp` algorithm from the LICORS package is unfortunately quite computational 
and is in principle not recommended for large grids / to generate large number of 
samples. Instead, we recommend clustering feature space using the `h2o.kmeans` function 
from the [h2o 
package](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/k-means.html), 
which is also suitable for larger datasets with computing running in parallel:


```r
library(h2o)
h2o.init(nthreads = -1)
#> 
#> H2O is not running yet, starting it now...
#> 
#> Note:  In case of errors look at the following log files:
#>     /tmp/RtmpJi5Csw/file52ef4b12809c/h2o_tomislav_started_from_r.out
#>     /tmp/RtmpJi5Csw/file52ef3589a9d5/h2o_tomislav_started_from_r.err
#> 
#> 
#> Starting H2O JVM and connecting: .. Connection successful!
#> 
#> R is connected to the H2O cluster: 
#>     H2O cluster uptime:         2 seconds 68 milliseconds 
#>     H2O cluster timezone:       Europe/Amsterdam 
#>     H2O data parsing timezone:  UTC 
#>     H2O cluster version:        3.30.0.1 
#>     H2O cluster version age:    1 year, 9 months and 17 days !!! 
#>     H2O cluster name:           H2O_started_from_R_tomislav_vru837 
#>     H2O cluster total nodes:    1 
#>     H2O cluster total memory:   15.71 GB 
#>     H2O cluster total cores:    32 
#>     H2O cluster allowed cores:  32 
#>     H2O cluster healthy:        TRUE 
#>     H2O Connection ip:          localhost 
#>     H2O Connection port:        54321 
#>     H2O Connection proxy:       NA 
#>     H2O Internal Security:      FALSE 
#>     H2O API Extensions:         Amazon S3, XGBoost, Algos, AutoML, Core V3, TargetEncoder, Core V4 
#>     R Version:                  R version 4.0.2 (2020-06-22)
#> Warning in h2o.clusterInfo(): 
#> Your H2O cluster version is too old (1 year, 9 months and 17 days)!
#> Please download and install the latest version from http://h2o.ai/download/
df.hex <- as.h2o(eberg_spc@predicted@data[,1:4], destination_frame = "df")
#>   |                                                                              |                                                                      |   0%  |                                                                              |======================================================================| 100%
km.nut <- h2o.kmeans(training_frame=df.hex, k=100, keep_cross_validation_predictions = TRUE)
#>   |                                                                              |                                                                      |   0%  |                                                                              |=================================================                     |  70%  |                                                                              |======================================================================| 100%
#km.nut
```

Note: in the example above, we have manually set the number of clusters to 100, 
but the number of clusters could be also derived using [some optimization procedure](https://www.r-bloggers.com/2019/01/10-tips-for-choosing-the-optimal-number-of-clusters/). 
Next, we predict the clusters and plot the output:


```r
m.km <- as.data.frame(h2o.predict(km.nut, df.hex, na.action=na.pass))
#>   |                                                                              |                                                                      |   0%  |                                                                              |======================================================================| 100%
```

We can save the class centers (if needed for any further analysis):


```r
class_df.c = as.data.frame(h2o.centers(km.nut))
names(class_df.c) = names(eberg_spc@predicted@data[,1:4])
str(class_df.c)
#> 'data.frame':	100 obs. of  4 variables:
#>  $ PC1: num  1.77 -1.88 -2.16 1.48 -6.14 ...
#>  $ PC2: num  -0.0139 2.5271 -4.9895 2.2981 0.6775 ...
#>  $ PC3: num  -1.332 5.526 -3.361 3.143 -0.605 ...
#>  $ PC4: num  0.0427 4.5728 -2.9965 -2.2581 1.5728 ...
#write.csv(class_df.c, "NCluster_100_class_centers.csv")
```

To select sampling points we use the minimum distance to class centers [@Brus2021sampling]:


```r
D <- fields::rdist(x1 = class_df.c, x2 = eberg_spc@predicted@data[,1:4])
units <- apply(D, MARGIN = 1, FUN = which.min)
rnd.fscs <- eberg_spc@predicted@coords[units,]
```


```r
plot(raster(eberg_spc@predicted[1]), col=SAGA_pal[[1]])
points(rnd.fscs, pch="+")
```

<div class="figure" style="text-align: center">
<img src="sampling_files/figure-html/eberg-fscs-1.png" alt="An example of a Feature Space Coverage Sampling (FSCS)." width="80%" />
<p class="caption">(\#fig:eberg-fscs)An example of a Feature Space Coverage Sampling (FSCS).</p>
</div>

Visually, FSCS seem to add higher spatial density of points in areas where there 
is higher complexity. The `h2o.kmeans` algorithm stratifies area into most 
possible homogeneous units (in the example above, large plains in the right 
part of the study area are relatively homogeneous, hence the sampling intensity 
in there areas drops significantly when visualized in the geographical space), 
and the points are then allocated per each strata.


```r
p <- plot(hb, colramp = reds, main='PCA Ebergotzen FSCS')
pushHexport(p$plot.vp)
grid.points(eberg_spc@predicted@data[units,"PC1"], eberg_spc@predicted@data[units,"PC2"], pch="+")
```

<div class="figure" style="text-align: center">
<img src="sampling_files/figure-html/eberg-fscs-pnts-1.png" alt="Distribution of the FSCS points from the previous example displayed in the feature space." width="80%" />
<p class="caption">(\#fig:eberg-fscs-pnts)Distribution of the FSCS points from the previous example displayed in the feature space.</p>
</div>

The FSCS sampling pattern in feature space looks almost as grid sampling in feature 
space. FSCS seems to put more effort on sampling at the edges on the feature space 
in comparison to LHS and SRS, and hence can be compared to classical response 
surface designs such as [D-optimal designs](https://en.wikipedia.org/wiki/Optimal_design) [@Hengl2004AJSR].


```r
h2o.shutdown(prompt = FALSE)
```

## Summary points

In the previous examples we have shown differences between SRS, LHS and FSCS.
SRS and LHS are IID sampling methods and as long as the number of samples is large 
and the study area is complex, it is often difficult to notice or quantify any under- or 
over-sampling. Depending on which variant of FSCS we implement, FSCS can result in 
higher spatial spreading especially if a study area consists of a combination 
of a relatively homogeneous and complex terrain units.

The Ebergotzen dataset (existing point samples) clearly shows that the _“convenience surveys”_ can 
show significant clustering and under-representation of feature space 
(Fig. \@ref(fig:eberg-fs2)). Consequently, these sampling bias could lead to:

- _Bias in estimating regression parameters_ i.e. over-fitting;  
- _Extrapolation problems_ due to under-representation of feature space;  
- _Bias_ in estimating population parameters of the target variable;  

The first step to deal with these problems is to detect them, second is to try 
to implement a strategy that prevents from model over-fitting. Some possible approaches 
to deal with such problems are addressed in the second part of the tutorial.

LHS and FSCS are recommended sampling methods if the purpose of sampling is to 
build regression or classification models using multitude of (terrain, 
climate, land cover etc) covariate layers. A generalization of LHS is the balanced 
sampling where users can select even variable inclusion probabilities [@grafstrom2014efficient; @brus2015balanced].

@ma2020comparison compared LHS to FSCS 
for mapping soil types and concluded that FSCS results in better mapping accuracy, 
most likely because FSCS spreads points better in feature space and hence in 
their case studies that seem to have helped with producing more accurate predictions.
@yang2020evaluation also report that LHS helps improve accuracy only for the large size of points.

The `h2o.kmeans` algorithm is suited for large datasets, but nevertheless to 
generate ≫100 clusters using large number of raster layers could become RAM 
consuming and is maybe not practical for operational sampling. An alternative 
would be to reduce number of clusters and select multiple points per cluster.

In the case of doubt which method to use LHS or FSCS, we recommend the following 
simple rules of thumb: 

- If your dataset contains relatively smaller-size rasters and the targeted number of sampling 
   points is relatively small (e.g. ≪1000), we recommend using the FSCS algorithm;  
- If your project requires large number of sampling points (≫100), then you should probably 
  consider using the LHS algorithm;  

In general, as the number of sampling points starts growing, differences between 
SRS (no feature space) and LHS becomes minor, which can also be witnessed 
visually (basically it becomes difficult to tell the difference between the two). 
SRS could, however, by accident miss some important parts of the feature space, 
so in principle it is still important to use either LHS or FSCS algorithms to 
prepare sampling locations for ML where objective is to fit regression and/or 
classification models using ML algorithms.

To evaluate potential sampling clustering and pin-point under-represented areas 
one can run multiple diagnostics:

1. In _geographical space_: 
   (a) [kernel density analysis](https://rdrr.io/cran/spatstat.core/man/density.ppp.html) using the spatstat package, then determine if some parts of the study area have systematically higher density;  
   (b) testing for **Complete Spatial Randomness** [@schabenberger2005statistical] using e.g. [spatstat.core::mad.test](https://rdrr.io/cran/spatstat.core/man/dclf.test.html) and/or [dbmss::Ktest](https://rdrr.io/cran/dbmss/man/Ktest.html);  
2. In _feature space_: 
   (a) occurrence probability analysis using the [maxlike package](https://rdrr.io/github/rbchan/maxlike/man/maxlike-package.html);
   (b) **unsupervised clustering** of the feature space using e.g. `h2o.kmeans`, then determining if 
   any of the clusters are significantly under-represented / under-sampled;  
   (c) estimating **Area of Applicability** based on similarities between training 
   prediction and feature spaces [@meyer2021predicting];  

Plotting generated sampling points both on a map and _feature space map_ helps 
detect possible extrapolation problems in a sampling design (Fig. \@ref(fig:eberg-fs2)). 
If you detect problems in feature space representation based on an existing point 
sampling set, you can try to reduce those problems by adding additional samples e.g. through 
**covariate space infill sampling** [@Brus2021sampling] or through 2nd round 
sampling and then re-analysis. Such methods are discussed in further chapters.
