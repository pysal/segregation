# Segregation Analysis, Inference, and Decomposition with PySAL
[![Build Status](https://travis-ci.com/pysal/segregation.svg?branch=master)](https://travis-ci.org/pysal/segregation)

![](doc/_static/images/heatmaps.png)

The PySAL `segregation` package is a tool for analyzing patterns of urban segregation.
With only a few lines of code, `segregation` users can:

Calculate over 40 segregation measures from simple to state-of-the art, including:

- [aspatial measures](https://github.com/pysal/segregation/blob/master/notebooks/aspatial_examples.ipynb)
- spatial measures
  - [using euclidian distances](https://github.com/pysal/segregation/blob/master/notebooks/spatial_examples.ipynb)
  - [using street network distances](https://github.com/pysal/segregation/blob/master/notebooks/network_measures.ipynb)
  - [using multiscalar definitions](https://github.com/pysal/segregation/blob/master/notebooks/multiscalar_segregation_profiles.ipynb)
- [local measures](https://github.com/pysal/segregation/blob/master/notebooks/local_measures_example.ipynb)

Test whether segregation estimates are statistically significant:

- [single value inference](https://github.com/pysal/segregation/blob/master/notebooks/inference_wrappers_example.ipynb)
- [comparative inference](https://github.com/pysal/segregation/blob/master/notebooks/inference_wrappers_example.ipynb)

[Decompose](https://github.com/pysal/segregation/blob/master/notebooks/decomposition_wrapper_example.ipynb)
segregation comparisons into

- differences arising from spatial structure 
- differences arising from demographic structure


## Installation
Released versions of segregation are available on pip and anaconda

pip:
```bash
pip install segregation
```

[anaconda](https://www.anaconda.com/download/):
```bash
conda install -c conda-forge segregation
```


You can also install the current development version from this repository 

 download [anaconda](https://www.anaconda.com/download/):

`cd` into the directory and run the following commands
```bash
conda env create -f environment.yml
conda activate segregation
python setup.py develop
```


## Getting started

For a complete guide to the `segregation` API, see the online [documentation](http://segregation.readthedocs.io).  

For code walkthroughs and sample analyses, see the [example notebooks](https://github.com/pysal/segregation/tree/master/notebooks)

### Single group measures

Each index in the **segregation** module is implemented as a class, which is built from a `pandas.DataFrame` or a `geopandas.GeoDataFrame`.
To estimate a segregation statistic, a user needs to call the segregation class she wishes to estimate, and pass three arguments:
- the DataFrame containing population data
- the name of the column with population totals for the group of interest
- the name of the column with the total population for each spatial unit

If, for example, a user would want to fit a dissimilarity index (D) to a DataFrame
called `df` to a specific group with 

frequency <tt>freq</tt> with each total population <tt>
population</tt>, a usual call would be something like this:

```python
from segregation.aspatial import Dissim
d_index = Dissim(df, "freq", "population")
```

If a user would want to fit a spatial dissimilarity index (SD) to a geopandas DataFrame
called `gdf` to a specific group with frequency `freq` with each total population 
`population`, a usual call would be something like this:

```python
from segregation.spatial import SpatialDissim
spatial_index = SpatialDissim(gdf, "freq", "population")
```

Every class in **segregation** has a `statistic` and a `core_data` attributes.
The first is a direct access to the point estimation of the specific segregation measure
and the second attribute gives access to the main data that the module uses internally to
perform the estimates.
To see the estimated D in the first generic example above, the user would have just to run
`index.statistic` to see the fitted value.

For point estimation, all the measures available can be summarized in the following table:

| **Measure**                                       | **Class/Function**              | **Spatial?** | **Specific Arguments** |
|:--------------------------------------------------|:--------------------------------|:------------:|:----------------------:|
| Dissimilarity (D)                                 | Dissim                          |      No      |           -            |
| Gini (G)                                          | GiniSeg                         |      No      |           -            |
| Entropy (H)                                       | Entropy                         |      No      |           -            |
| Isolation (xPx)                                   | Isolation                       |      No      |           -            |
| Exposure (xPy)                                    | Exposure                        |      No      |           -            |
| Atkinson (A)                                      | Atkinson                        |      No      |           b            |
| Correlation Ratio (V)                             | CorrelationR                    |      No      |           -            |
| Concentration Profile (R)                         | ConProf                         |      No      |           m            |
| Modified Dissimilarity (Dct)                      | ModifiedDissim                  |      No      |       iterations       |
| Modified Gini (Gct)                               | ModifiedGiniSeg                 |      No      |       iterations       |
| Bias-Corrected Dissimilarity (Dbc)                | BiasCorrectedDissim             |      No      |           B            |
| Density-Corrected Dissimilarity (Ddc)             | DensityCorrectedDissim          |      No      |          xtol          |
| Spatial Proximity Profile (SPP)                   | SpatialProxProf                 |     Yes      |           m            |
| Spatial Dissimilarity (SD)                        | SpatialDissim                   |     Yes      |     w, standardize     |
| Boundary Spatial Dissimilarity (BSD)              | BoundarySpatialDissim           |     Yes      |      standardize       |
| Perimeter Area Ratio Spatial Dissimilarity (PARD) | PerimeterAreaRatioSpatialDissim |     Yes      |      standardize       |
| Distance Decay Isolation (DDxPx)                  | DistanceDecayIsolation          |     Yes      |      alpha, beta       |
| Distance Decay Exposure (DDxPy)                   | DistanceDecayExposure           |     Yes      |      alpha, beta       |
| Spatial Proximity (SP)                            | SpatialProximity                |     Yes      |      alpha, beta       |
| Absolute Clustering (ACL)                         | AbsoluteClustering              |     Yes      |      alpha, beta       |
| Relative Clustering (RCL)                         | RelativeClustering              |     Yes      |      alpha, beta       |
| Delta (DEL)                                       | Delta                           |     Yes      |           -            |
| Absolute Concentration (ACO)                      | AbsoluteConcentration           |     Yes      |           -            |
| Relative Concentration (RCO)                      | RelativeConcentration           |     Yes      |           -            |
| Absolute Centralization (ACE)                     | AbsoluteCentralization          |     Yes      |           -            |
| Relative Centralization (RCE)                     | RelativeCentralization          |     Yes      |           -            |

### Multigroup measures

It also possible to estimate Multigroup measures.
This framework also relies on [pandas](https://github.com/pandas-dev/pandas) DataFrames for
the aspatial measures.

Suppose you have a DataFrame called <tt>df</tt> that has populations of some groups, for example,
`Group A`, `Group B` and <tt>Group C</tt>. A usual call for a multigroup Dissimilarity index would
be:

```python
from segregation.aspatial import MultiDissim
index = MultiDissim(df, ['Group A', 'Group B', 'Group C'])
```

Therefore, a `statistic` attribute will contain the value of this index.

Currently, theses indexes are summarized in the table below:

| **Measure**                                 | **Class/Function**               | **Spatial?** | **Specific Arguments** |
|:--------------------------------------------|:---------------------------------|:------------:|:----------------------:|
| Multigroup Dissimilarity                    | MultiDissim                      |      No      |           -            |
| Multigroup Gini                             | MultiGiniSeg                     |      No      |           -            |
| Multigroup Normalized Exposure              | MultiNormalizedExposure          |      No      |           -            |
| Multigroup Information Theory               | MultiInformationTheory           |      No      |           -            |
| Multigroup Relative Diversity               | MultiRelativeDiversity           |      No      |           -            |
| Multigroup Squared Coefficient of Variation | MultiSquaredCoefficientVariation |      No      |           -            |
| Multigroup Diversity                        | MultiDiversity                   |      No      |       normalized       |
| Simpson’s Concentration                     | SimpsonsConcentration            |      No      |           -            |
| Simpson’s Interaction                       | SimpsonsInteraction              |      No      |           -            |
| Multigroup Divergence                       | MultiDivergence                  |      No      |           -            |

### Local measures

Also, it is possible to calculate local measures of segregation.
A <tt>statistics</tt> (the attribute is in the plural since, many statistics are fitted)
attribute will contain the values of these indexes.
Currently, they are summarized in the table below:

| **Measure**                   | **Class/Function**             | **Spatial?** | **Specific Arguments** |
|:------------------------------|:-------------------------------|:------------:|:----------------------:|
| Location Quotient             | MultiLocationQuotient          |      No      |           -            |
| Local Diversity               | MultiLocalDiversity            |      No      |           -            |
| Local Entropy                 | MultiLocalEntropy              |      No      |           -            |
| Local Simpson’s Concentration | MultiLocalSimpsonConcentration |      No      |           -            |
| Local Simpson’s Interaction   | MultiLocalSimpsonInteraction   |      No      |           -            |
| Local Centralization          | LocalRelativeCentralization    |     Yes      |           -            |

### Inference

Once the segregation indexes are fitted, the user can perform inference to shed light for
statistical significance in regional analysis.
The summary of the inference framework is presented in the table below:

| **Inference Type** | **Class/Function** |                    **Function main Inputs**                    |       **Function Outputs**       |
|:-------------------|:-------------------|:--------------------------------------------------------------:|:--------------------------------:|
| Single Value       | SingleValueTest    |  seg_class, iterations_under_null, null_approach, two_tailed   |   p_value, est_sim, statistic    |
| Two Values         | TwoValueTest       | seg_class_1, seg_class_2, iterations_under_null, null_approach | p_value, est_sim, est_point_diff |

Another useful analytics that can be performed with the **segregation** module is a
decompositional approach where two different indexes can be brake down into spatial
components (`c_s`) and attribute component (`c_a`). This framework is summarized in the
table below:

### Decomposition
| **Framework** | **Class/Function**   |        **Function main Inputs**         | **Function Outputs** |
|:--------------|:---------------------|:---------------------------------------:|:--------------------:|
| Decomposition | DecomposeSegregation | index1, index2, counterfactual_approach |       c_a, c_s       |

## Contributing

PySAL-segregation is under active development and contributors are welcome.

If you have any suggestion, feature request, or bug report, please open a new
[issue](https://github.com/pysal/inequality/issues) on GitHub.
To submit patches, please follow the PySAL development
[guidelines](http://pysal.readthedocs.io/en/latest/developers/index.html) and open a
[pull request](https://github.com/pysal/segregation). Once your changes get merged, you’ll
automatically be added to the
[Contributors List](https://github.com/pysal/segregation/graphs/contributors).

## Support

If you are having issues, please talk to us in the
[gitter room](https://gitter.im/pysal/pysal).

## License

The project is licensed under the
[BSD license](https://github.com/pysal/pysal/blob/master/LICENSE.txt).

## Funding

<img src="figs/nsf_logo.jpg" width="50"> Award #1831615
[RIDIR: Scalable Geospatial Analytics for Social Science Research](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1831615)

<img src="figs/capes_logo.jpg" width="50"> Renan Xavier Cortes is grateful for the support of Coordenação de Aperfeiçoamento de
Pessoal de Nível Superior - Brazil (CAPES) - Process number 88881.170553/2018-01
