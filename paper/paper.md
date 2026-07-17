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

Although segregation analysis is also available in R packages such as `OasisR` [@tivadar2019oasisr], `seg` [@segrhong2011], and `segregation` [@Elbers2021], PySAL's `segregation` module provides a broader and more integrated framework. It includes over 40 segregation indices, supports multiscalar and network-based analyses, Monte Carlo inference, and decomposition methods, while enabling simultaneous computation of multiple indices with minimal software dependencies. As a native Python library, it integrates seamlessly with the PySAL ecosystem and modern geospatial data science workflows.

# Software design

`segregation` is designed with attention to flexibility, usability, and interoperability between its core functions. The library is organized into different subparts such as `singlegroup`, `multigroup`, `local`, `inference`, `decomposition`, `batch`, `network`, and `dynamics`.[^1] Segregation measures are built using Python classes which can be integrated in subsequent steps, such as inference or decomposition. The module is structured toward two kinds of segregation indices: 'spatially-explicit' and 'spatially-implicit'. The former includes space as part of its original formula. The latter uses the @reardonsullivan2004 approach to state that any segregation index is a spatial index if you transform the data properly.

[^1]: @cortes2020open introduced `segregation` but many new implementations were developed recently and the API underwent a major revision.

Additionally, `segregation` is developed with testing and documentation standards consistent with the Scientific Python ecosystem, ensuring reliability and maintainability.

<!-- "scikit-learn" like API? -->

## Core Functionality

`segregation` organizes its functionality around the type of segregation analysis the user is interested in, and each subpart is explained as follows.


### Single and Multigroup Indices

Single-group measures assess segregation between two different groups in a given location (i.e., one group vs. everyone else). Multigroup segregation evaluates the simultaneous separation of all groups in a population (e.g., the distribution of White, Black, Asian, and Hispanic residents) across areas. 

Currently, `segregation` has over 40 indices available which represents, to our knowledge, the broadest range of indices available for a user in any software. Also, the user can fit many indices at once with a wrapper function in the `batch` module.

### Local Indices

Unlike global indices that summarize an entire metropolitan area into a single value, local indices decompose segregation to the individual geographic unit level. Using these disaggregated measures helps identify precise spatial clusters where social isolation is most acute, uncovering micro-level dynamics that global metrics often mask. Currently, `segregation` has seven local indexes. 

### Multiscalar

The multiscalar profile [@reardon2008geographic] is a tool for measuring spatial segregation dynamics--the way that a segregation index changes values as the concept of a neighborhood changes, and what that tells us about macro versus micro patterns of segregation. The core idea is to calculate a segregation statistic, then expand the spatial scope of a neighborhood, recalculate the statistic, and repeat.

The package has a wrapper named `compute_multiscalar_profile` which can be used in a workflow to build these profiles.

### Simulation based Inference

PySAL's `segregation` module provides Monte Carlo inference for evaluating the statistical significance of segregation indices under different null hypotheses. For single-value inference, it supports resampling approaches such as `bootstrap`, `systematic`, `evenness`, and `geographic_permutation`. For comparative inference, it includes methods such as `bootstrap` and `composition`, which generate synthetic distributions through counterfactual estimates. Because different null hypotheses test distinct assumptions, their specification is critical and can lead to substantially different conclusions. Likewise, not all segregation indices are appropriate for every null hypothesis, particularly in comparative analyses, making careful selection of both the index and inference procedure essential.


### Decomposition

The PySAL `segregation` module implements a decomposition framework for comparative segregation analysis that partitions differences in segregation into population composition and spatial structure. Building on @rey2021comparative, it combines spatially explicit counterfactual distributions with Shapley value decomposition to quantify each component's contribution to differences in segregation across cities, time periods, and spatiotemporal contexts. Applicable to multiple segregation indices, the framework provides a richer interpretation than direct index comparisons by identifying whether observed differences primarily reflect demographic composition or neighborhood spatial organization. This functionality is available through the `DecomposeSegregation` class in the `decomposition` subpart.

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
