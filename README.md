# Segregation Analysis, Inference, and Decomposition with PySAL

[![codecov](https://codecov.io/gh/pysal/segregation/branch/master/graph/badge.svg?token=1ujvZCI9Ce)](https://codecov.io/gh/pysal/segregation)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/segregation)
![PyPI](https://img.shields.io/pypi/v/segregation)
![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/segregation)
![GitHub commits since latest release (branch)](https://img.shields.io/github/commits-since/pysal/segregation/latest)
[![DOI](https://zenodo.org/badge/162503796.svg)](https://zenodo.org/badge/latestdoi/162503796)
[![Documentation](https://img.shields.io/static/v1.svg?label=docs&message=current&color=9cf)](http://pysal.org/segregation/)


![](doc/_static/images/heatmaps.png)

The PySAL **segregation** package is a tool for analyzing patterns of urban segregation.
With only a few lines of code, **segregation** users can

Calculate over 40 segregation measures from simple to state-of-the art, including:

- [aspatial segregation indices](https://github.com/pysal/segregation/blob/master/notebooks/aspatial_examples.ipynb)
- spatial segregation indices
  - [using spatial weights matrices, euclidian distances, or topological relationships](https://github.com/pysal/segregation/blob/master/notebooks/spatial_examples.ipynb)
  - [using street network distances](https://github.com/pysal/segregation/blob/master/notebooks/network_measures.ipynb)
  - [using multiscalar definitions](https://github.com/pysal/segregation/blob/master/notebooks/multiscalar_segregation_profiles.ipynb)
- [local segregation indices](https://github.com/pysal/segregation/blob/master/notebooks/local_measures_example.ipynb)

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

For a complete guide to the `segregation` API, see the online
[documentation](https://pysal.org/segregation/). 

For code walkthroughs and sample analyses, see the
[example notebooks](https://github.com/pysal/segregation/tree/master/notebooks)

## Calculating Segregation Measures

Each index in the **segregation** module is implemented as a class, which is built from a `pandas.DataFrame`
or a `geopandas.GeoDataFrame`. To estimate a segregation statistic, a user needs to call the segregation class
she wishes to estimate, and pass three arguments:

- the DataFrame containing population data
- the name of the column with population counts for the group of interest
- the name of the column with the total population for each enumeration unit

Every class in **segregation** has a `statistic` and a `core_data` attributes.
The first is a direct access to the point estimation of the specific segregation measure
and the second attribute gives access to the main data that the module uses internally to
perform the estimates.

### Single group measures

If, for example, a user was studying income segregation and wanted to know whether
high-income residents tend to be more segregated from others.
This user may want would want to fit a dissimilarity index (D) to a DataFrame called `df` to
a specific group with columns like `"hi_income"`, `"med_income"` and `"low_income"` that store counts of people in each income
bracket, and a total column called `"total_population"`. A typical call would be something like this:

```python
from segregation.aspatial import Dissim
d_index = Dissim(df, "hi_income", "total_population")
```

To see the estimated D in the first generic example above, the user would have just to run
`d_index.statistic` to see the fitted value.

If a user would want to fit a *spatial* dissimilarity index (SD), the call would be nearly
identical, save for the fact that the `DataFrame` now needs to be a `GeoDataFrame` with an appropriate `geometry` column

```python
from segregation.spatial import SpatialDissim
spatial_index = SpatialDissim(gdf, "hi_income", "total_population")
```

Some spatial indices can also accept either a [PySAL](http://pysal.org) `W` object, or a [pandana](https://github.com/UDST/pandana) `Network` object,
which allows the user full control over how to parameterize spatial effects.
The network functions can be particularly useful for teasing out differences in
segregation measures caused by two cities that have two very different spatial structures,
like for example Detroit MI (left) and Monroe LA (right):

![](doc/_static/images/networks.png)

For point estimation, all single-group indices available are summarized in the following
table:

| **Measure**                                       | **Class/Function**              | **Spatial?** |    **Specific Arguments**      |
|:--------------------------------------------------|:--------------------------------|:------------:|:-----------------------------: |
| Dissimilarity (D)                                 | Dissim                          |      No      |           -                    |
| Gini (G)                                          | GiniSeg                         |      No      |           -                    |
| Entropy (H)                                       | Entropy                         |      No      |           -                    |
| Isolation (xPx)                                   | Isolation                       |      No      |           -                    |
| Exposure (xPy)                                    | Exposure                        |      No      |           -                    |
| Atkinson (A)                                      | Atkinson                        |      No      |           b                    |
| Correlation Ratio (V)                             | CorrelationR                    |      No      |           -                    |
| Concentration Profile (R)                         | ConProf                         |      No      |           m                    |
| Modified Dissimilarity (Dct)                      | ModifiedDissim                  |      No      |       iterations               |
| Modified Gini (Gct)                               | ModifiedGiniSeg                 |      No      |       iterations               |
| Bias-Corrected Dissimilarity (Dbc)                | BiasCorrectedDissim             |      No      |           B                    |
| Density-Corrected Dissimilarity (Ddc)             | DensityCorrectedDissim          |      No      |          xtol                  |
| Minimun-Maximum Index (MM)                        | MinMax                          |      No      |                                |
| Spatial Proximity Profile (SPP)                   | SpatialProxProf                 |     Yes      |           m                    |
| Spatial Dissimilarity (SD)                        | SpatialDissim                   |     Yes      |     w, standardize             |
| Boundary Spatial Dissimilarity (BSD)              | BoundarySpatialDissim           |     Yes      |      standardize               |
| Perimeter Area Ratio Spatial Dissimilarity (PARD) | PerimeterAreaRatioSpatialDissim |     Yes      |      standardize               |
| Distance Decay Isolation (DDxPx)                  | DistanceDecayIsolation          |     Yes      |      alpha, beta, metric       |
| Distance Decay Exposure (DDxPy)                   | DistanceDecayExposure           |     Yes      |      alpha, beta, metric       |
| Spatial Proximity (SP)                            | SpatialProximity                |     Yes      |      alpha, beta, metric       |
| Absolute Clustering (ACL)                         | AbsoluteClustering              |     Yes      |      alpha, beta, metric       |
| Relative Clustering (RCL)                         | RelativeClustering              |     Yes      |      alpha, beta, metric       |
| Delta (DEL)                                       | Delta                           |     Yes      |           -                    |
| Absolute Concentration (ACO)                      | AbsoluteConcentration           |     Yes      |           -                    |
| Relative Concentration (RCO)                      | RelativeConcentration           |     Yes      |           -                    |
| Absolute Centralization (ACE)                     | AbsoluteCentralization          |     Yes      |           -                    |
| Relative Centralization (RCE)                     | RelativeCentralization          |     Yes      |           -                    |
| Relative Centralization (RCE)                     | RelativeCentralization          |     Yes      |           -                    |
| Spatial Minimun-Maximum (SMM)                     | SpatialMinMax                   |     Yes      | network, w, decay, distance, precompute |

### Multigroup measures

**segregation** also facilitates the estimation of multigroup segregation measures.

In this case, the call is nearly identical to the single-group, only now we pass a list of
column names rather than a single string;
reprising the income segregation example above, an example call might look like this  

```python
from segregation.aspatial import MultiDissim
index = MultiDissim(df, ['hi_income', 'med_income', 'low_income'])
```

```python
index.statistic
```

Available multi-group indices are summarized in the table below:

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
A `statistics` attribute will contain the values of these indexes. **Note:
in this case the attribute is in the plural since, many statistics are fitted, one for
each enumeration unit** Local segregation indices have the same signature as their global
cousins and are summarized in the table below:

| **Measure**                   | **Class/Function**             | **Spatial?** | **Specific Arguments** |
|:------------------------------|:-------------------------------|:------------:|:----------------------:|
| Location Quotient             | MultiLocationQuotient          |      No      |           -            |
| Local Diversity               | MultiLocalDiversity            |      No      |           -            |
| Local Entropy                 | MultiLocalEntropy              |      No      |           -            |
| Local Simpson’s Concentration | MultiLocalSimpsonConcentration |      No      |           -            |
| Local Simpson’s Interaction   | MultiLocalSimpsonInteraction   |      No      |           -            |
| Local Centralization          | LocalRelativeCentralization    |     Yes      |           -            |

## Testing for Statistical Significance

Once the segregation indexes are fitted, the user can perform inference to shed light for
statistical significance in regional analysis.
The summary of the inference framework is presented in the table below:

| **Inference Type** | **Class/Function** |                    **Function main Inputs**                    |       **Function Outputs**       |
|:-------------------|:-------------------|:--------------------------------------------------------------:|:--------------------------------:|
| Single Value       | SingleValueTest    |  seg_class, iterations_under_null, null_approach, two_tailed   |   p_value, est_sim, statistic    |
| Two Values         | TwoValueTest       | seg_class_1, seg_class_2, iterations_under_null, null_approach | p_value, est_sim, est_point_diff |

### [Single Value Inference](https://github.com/pysal/segregation/blob/master/notebooks/inference_wrappers_example.ipynb)

![](doc/_static/images/singleval_inference.png)

### [Two-Value Inference](https://github.com/pysal/segregation/blob/master/notebooks/inference_wrappers_example.ipynb)

![](doc/_static/images/twoval_inference.png)

### [Decomposition](https://github.com/pysal/segregation/blob/master/notebooks/decomposition_wrapper_example.ipynb)

Another useful analysis that can be performed with the **segregation** module is a
decompositional approach where two different indexes can be broken down into their spatial
component (`c_s`) and attribute component (`c_a`). This framework is summarized in the table
below:

| **Framework** | **Class/Function**   |        **Function main Inputs**         | **Function Outputs** |
|:--------------|:---------------------|:---------------------------------------:|:--------------------:|
| Decomposition | DecomposeSegregation | index1, index2, counterfactual_approach |       c_a, c_s       |

![](doc/_static/images/decomp_example.png)

In this case, the difference in measured D statistics between Detroit and Monroe is
attributable primarily to their demographic makeup, rather than the spatial structure of
the two cities.
(Note, this is to be expected since *D* is not a spatial index)

## Contributing

PySAL-segregation is under active development and contributors are welcome.

If you have any suggestion, feature request, or bug report, please open a new
[issue](https://github.com/pysal/segregation/issues) on GitHub.
To submit patches, please follow the PySAL development
[guidelines](https://github.com/pysal/pysal/wiki) and open a
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

## Citation
To cite `segregation`, we recommend the following

```latex
@software{renan_xavier_cortes_2020,
  author       = {Renan Xavier Cortes and
                  eli knaap and
                  Sergio Rey and
                  Wei Kang and
                  Philip Stephens and
                  James Gaboardi and
                  Levi John Wolf and
                  Antti Härkönen and
                  Dani Arribas-Bel},
  title        = {PySAL/segregation: Segregation Analysis, Inference, & Decomposition},
  month        = feb,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3265359},
  url          = {https://doi.org/10.5281/zenodo.3265359}
}
```
