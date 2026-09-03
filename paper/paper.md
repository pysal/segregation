---
title: 'segregation: Segregation Analysis, Inference, and Decomposition in Python'
tags:
- Python
- geographic information science
- segregation
- spatial analysis
date: "09 June 2026"
output: pdf_document
authors:
- name: Renan X. Cortes
  orcid: "0000-0002-1889-5282"
  equal-contrib: true
  affiliation: 1
- name: Elijah Knaap
  orcid: "0000-0001-7520-2238"
  equal-contrib: true
  affiliation: 2
- name: Sergio J. Rey
  orcid: "0000-0001-5857-9762"
  equal-contrib: true
  affiliation: 3
bibliography: paper.bib
affiliations:
- name: Federal University of Rio Grande do Sul
  index: 1
- name: University of California, Irvine
  index: 2
- name: San Diego State University
  index: 3
---

<!--

NEW JOSS REQUIREMENTS

Length

* longer paper acceptable (but not necessarily encouraged...)
*  750-1750

Sections

https://joss.readthedocs.io/en/latest/paper.html#:~:text=Your%20paper%20must-,include,-the%20following%20required

* Summary
* Statement of need
* State of the field
* Software design
* Research impact statement
* AI usage disclosure

-->


# Summary


`segregation` is an open-source Python library for measuring, analyzing, inferring, and decomposing social and spatial segregation. Developed within the PySAL ecosystem, it provides a comprehensive framework for quantifying segregation across demographic, socioeconomic, and geographic contexts using both aspatial and spatially explicit measures. The package implements over 40 single-group, multigroup, local, and spatial segregation indices, capturing dimensions such as evenness, exposure, concentration, clustering, centralization, and spatial proximity. It supports multiple spatial representations, including adjacency, distance-based, and network-based relationships.

Beyond point estimation, `segregation` provides Monte Carlo inference for single-value and comparative hypothesis testing under alternative null models, as well as decomposition methods that separate differences in segregation into demographic composition and spatial structure. Built for Pandas and GeoPandas, `segregation` integrates seamlessly with the PySAL ecosystem [@pysal2007; @rey2022pysalecosystem], providing transparent, reproducible, and extensible tools for segregation research and spatial data science.


# Statement of need

Residential segregation is a central topic in urban studies, sociology, demography, geography, and public policy. Although demographic and spatial data are increasingly available, existing software often provides a limited set of segregation measures, relies on proprietary platforms, or lacks support for modern inference and decomposition methods. Within Python, segregation analysis has historically depended on custom implementations despite the widespread adoption of Pandas and GeoPandas.

`segregation` addresses these limitations by providing:

* A comprehensive, well-documented API for segregation analysis
* Native integration with Pandas, GeoPandas, and the PySAL ecosystem
* Over 40 aspatial, spatial, local, and multigroup segregation indices
* Multiscalar segregation profiles
* Simulation-based inference for hypothesis testing
* Decomposition of segregation into demographic and spatial components
* Transparent, reproducible, and extensible implementations

These capabilities establish `segregation` as a unified, open-source framework for reproducible segregation analysis in Python.

# State of the field

`segregation` is part of the PySAL ecosystem, where it provides tools for residential segregation analysis within the **Explore** family of spatial analysis libraries. Unlike desktop GIS software, it supports reproducible, scriptable workflows that integrate naturally with broader Python data science pipelines.

The most closely related tools are the R packages `OasisR` [@tivadar2019oasisr], `seg` [@segrhong2011], and `segregation` [@Elbers2021]. `OasisR` provides a broad catalog of aspatial and spatial evenness indices with simulation-based inference; `seg` targets spatial evenness and exposure measures; and the R `segregation` package focuses on the entropy family (the Mutual Information index and Theil's *H*) with within/between and temporal decomposition and Bayesian bias correction. PySAL's `segregation` is distinguished less by any single index than by combining, in one class-based API, index families that these packages cover separately. These include single-group, multigroup, local, and explicitly spatial measures of evenness, exposure, concentration, clustering, and centralization, together with Monte Carlo inference under several null models, Shapley decomposition of differences into demographic and spatial components, multiscalar profiles, and batch computation. It depends only on the scientific-Python and PySAL stack (NumPy, pandas, GeoPandas, scikit-learn), with no proprietary or desktop-GIS requirements. A native Python implementation, rather than a wrapper around the R packages, keeps these workflows within the PySAL ecosystem and lets them share a common index interface.

# Software design

`segregation` is organized into subpackages by analysis task (`singlegroup`, `multigroup`, `local`, `inference`, `decomposition`, `batch`, `network`, and `dynamics`[^1]), so that the large collection of indices stays navigable and users import only what they need. Each index is implemented as a Python class rather than a bare function: fitting a class stores the input data, parameters, and result on one object, which `inference` and `decomposition` then consume directly. This estimator-style pattern follows a convention familiar from scikit-learn and avoids threading many arguments through a functional API. The design also distinguishes two kinds of indices. *Spatially-explicit* indices include space in their formula. *Spatially-implicit* indices instead accept spatial weights or a network and transform the data before estimation, following @reardonsullivan2004; one class therefore serves both aspatial and spatial use, at the cost of expressing spatial behavior through parameters rather than through distinct types.

[^1]: @cortes2020open introduced `segregation` but many new implementations were developed recently and the API underwent a major revision.

Additionally, `segregation` is developed with testing and documentation standards consistent with the Scientific Python ecosystem, ensuring reliability and maintainability.

## Core Functionality

`segregation` organizes its functionality around the type of segregation analysis the user is interested in, and each subpackage is explained as follows.


### Single and Multigroup Indices

Single-group measures assess segregation between two different groups in a given location (i.e., one group vs. everyone else). Multigroup segregation evaluates the simultaneous separation of all groups in a population (e.g., the distribution of White, Black, Asian, and Hispanic residents) across areas. 

Currently, `segregation` provides over 40 indices, which to our knowledge is among the most extensive selections in any segregation software, and the `batch` subpackage can fit many of them at once.

### Local Indices

Unlike global indices that summarize an entire metropolitan area into a single value, local indices decompose segregation to the individual geographic unit level. Using these disaggregated measures helps identify precise spatial clusters where social isolation is most acute, uncovering micro-level dynamics that global metrics often mask. Currently, `segregation` has seven local indices. 

### Multiscalar

The multiscalar profile [@reardon2008geographic] is a tool for measuring spatial segregation dynamics--the way that a segregation index changes values as the concept of a neighborhood changes, and what that tells us about macro versus micro patterns of segregation. The core idea is to calculate a segregation statistic, then expand the spatial scope of a neighborhood, recalculate the statistic, and repeat.

The package has a wrapper named `compute_multiscalar_profile` which can be used in a workflow to build these profiles.

### Simulation-based Inference

PySAL's `segregation` module provides Monte Carlo inference for evaluating the statistical significance of segregation indices under different null hypotheses. For single-value inference, it supports resampling approaches such as `bootstrap`, `systematic`, `evenness`, and `geographic_permutation`. For comparative inference, it includes methods such as `bootstrap` and `composition`, which generate synthetic distributions through counterfactual estimates. Because different null hypotheses test distinct assumptions, their specification is critical and can lead to substantially different conclusions. Likewise, not all segregation indices are appropriate for every null hypothesis, particularly in comparative analyses, making careful selection of both the index and inference procedure essential.


### Decomposition

The PySAL `segregation` module implements a decomposition framework for comparative segregation analysis that partitions differences in segregation into population composition and spatial structure. Building on @rey2021comparative, it combines spatially explicit counterfactual distributions with Shapley value decomposition to quantify each component's contribution to differences in segregation across cities, time periods, and spatiotemporal contexts. Applicable to multiple segregation indices, the framework provides a richer interpretation than direct index comparisons by identifying whether observed differences primarily reflect demographic composition or neighborhood spatial organization. This functionality is available through the `DecomposeSegregation` class in the `decomposition` subpackage.

## Example workflow

Assuming a `pandas` or `geopandas` DataFrame named `data_1`, a segregation index can be computed as:

```python
from segregation.singlegroup import Dissim

seg_index = Dissim(
    data_1,
    group_pop_var="group_A",
    total_pop_var="total_population",
)
```

The estimated value is available through the `statistic` attribute. Statistical significance can be evaluated using Monte Carlo inference:

```python
from segregation.inference import SingleValueTest

result = SingleValueTest(seg_index, null_approach="bootstrap")
```

The package also provides wrappers for computing all single-group indices (`batch_compute_singlegroup`), generating multiscalar profiles (`compute_multiscalar_profile`), comparing segregation measures (`TwoValueTest`), and decomposing differences into demographic and spatial components (`DecomposeSegregation`).



# Research impact statement

`segregation` is actively used in research on urban inequality, spatial demography, and education policy. Applications include comparative analyses of racial segregation [@cortes2020open], measurement of spatial information theory indices across U.S. metropolitan areas [@knaap2024segregated], studies of school-neighborhood segregation [@rey2024MeasuringSpatial], optimization approaches for reducing school segregation [@wei2022ReducingRacial], comparative segregation analytics [@rey2021comparative], and analyses of historical redlining legacies [@rey2022LegacyRedlining].

The package is also used in spatial data science education, including textbooks [@knaapUrbanAnalysis2026] and instructional materials presented at conferences such as SciPy.[^2]

[^2]: https://www.youtube.com/watch?v=4AHJVMs7iH4


# AI usage disclosure

No generative AI or LLMs were used for code development in `segregation`; however, they were used for grammar, spelling corrections and consistency during the writing of this paper.

# Acknowledgements

`segregation` is developed as part of the PySAL community, which brings together researchers and developers working on spatial analysis methods and software. The project builds on decades of research in segregation, urban, and spatial data science, and benefits from contributions across the open-source geospatial community.

Funding from National Science Foundation Grants [2345820](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2345820) and
[1831615](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=1831615&HistoricalAwards=false) have supported `segregation` development.

The following acknowledgement applies to Renan X. Cortes:

> Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) through process 88881.170553/2018-01 have supported `segregation` development.


# References
