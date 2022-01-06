---
knit: "bookdown::render_book"
title: "Spatial sampling and resampling for Machine Learning"
author: "Tom Hengl and Leandro Parente"
description: "This R tutorial contains instructions on how to organize spatial sampling using R packages. It is organized in three main parts: (1) planning new surveys: i.e. starting from scratch, (2) implementing resampling: learning from existing point data, focusing on subsampling and Cross-Validation strategies, (3) planning additional sampling: sampling additional point data based on initial models, the running re-analysis and gradually improving models until the maximum possible accuracy is reached. We use sample datasets to demonstrate processing steps and provide interpretation and dicussion of the results. More chapters will be added in the future. Contributions are welcome. To discuss issues or report a bug please use the repository homepage."
url: 'https\://opengeohub.github.io/spatial-sampling-ml/'
bibliography: ./tex/refs.bib
csl: ./tex/apa.csl  
fig_caption: yes
link-citations: yes
gihub-repo: OpenGeoHub/spatial-sampling-ml/
twitter-handle: opengeohub
cover-image: cover.png
site: bookdown::bookdown_site
documentclass: book
---



# Introduction {.unnumbered}

## Overview {.unnumbered}

[![Access source code](cover.png){.cover width="250"}](https://opengeohub.github.io/spatial-sampling-ml/) This [Rmarkdown tutorial](https://opengeohub.github.io/spatial-sampling-ml/) provides practical instructions, illustrated with sample 
dataset, on how to generate and evaluate sampling plans using your own data. 
The specific focus is put on preparing sampling designs for predictive mapping, 
running analysis and interpretation on existing point data and planning 2nd and 3rd 
round sampling (based on initial models).

We use several key R packages and existing tutorials including:

- [sp](https://github.com/edzer/sp) package,
- [clhs](https://github.com/pierreroudier/clhs) package,
- [mlr](https://mlr.mlr-org.com/) package,
- [ranger](https://github.com/imbs-hl/ranger/) package,
- [forestError](https://github.com/benjilu/forestError) package,

For an introduction to Spatial Data Science and Machine Learning with R we recommend studying first:

- Becker, M. et al.: **[“mlr3 book”](https://mlr3book.mlr-org.com/)**;  
- Irizarry, R.A.: **[“Introduction to Data Science: Data Analysis and Prediction Algorithms with R”](https://rafalab.github.io/dsbook/)**;  
- Molnar, C.: **[“Interpretable Machine Learning: A Guide for Making Black Box Models Explainable”](https://christophm.github.io/interpretable-ml-book/)**;  
- Lovelace, R., Nowosad, J. and Muenchow, J.: **[“Geocomputation with R”](https://geocompr.robinlovelace.net/)**;  
- Pebesma, E. and Bivand, R: **[“Spatial Data Science: with applications in R”](https://keen-swartz-3146c4.netlify.app/)**;  

If you are looking for a more gentle introduction to spatial sampling methods in R 
please refer to @Bivand2013Springer, @BRUS2019464 and @Brus2021sampling. 
The _“Spatial sampling with R”_ book by Dick Brus and R code examples are 
available via <https://github.com/DickBrus/SpatialSamplingwithR>.

For an introduction to **Predictive Soil Mapping** using R refer to <https://soilmapper.org>.

Machine Learning in **python** with resampling can be best implemented via the [scikit-learn library](https://scikit-learn.org/stable/), 
which matches in functionality what is available via the mlr package in R.

To install the most recent **landmap**, **ranger**, **forestError** and **clhs** packages from Github use:


```r
library(devtools)
devtools::install_github("envirometrix/landmap")
devtools::install_github("imbs-hl/ranger")
devtools::install_github("benjilu/forestError")
devtools::install_github("pierreroudier/clhs")
```

## License {.unnumbered}

[<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" />](http://creativecommons.org/licenses/by-sa/4.0/)

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).



## Acknowledgements {.unnumbered}

This tutorial is based on the **[“R for Data Science”](https://r4ds.had.co.nz/)** book by Hadley Wickham and contributors.

**[OpenLandMap](https://openlandmap.org)** is a collaborative effort and many people 
have contributed data, software, fixes and improvements via pull request. [OpenGeoHub](https://opengeohub.org) 
is an independent not-for-profit research foundation promoting Open Source and Open Data solutions. 
**[EnvirometriX Ltd.](https://envirometrix.nl)** is the commercial branch of the group 
responsible for designing soil sampling designs for the **[AgriCapture](https://agricaptureco2.eu/)** and similar soil monitoring projects.

[<img src="tex/opengeohub_logo_ml.png" alt="OpenGeoHub logo" width="350"/>](https://opengeohub.org)
