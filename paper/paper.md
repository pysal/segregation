---
title: 'segregation: Segregation Analysis, Inference, and Decomposition in Python'
tags:
- Python
- geographic information science
- segregation
- spatial analysis
date: "05 June 2026"
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


`segregation` is an open-source Python library for the measurement, analysis, inference, and decomposition of social and spatial segregation patterns. Developed as part of the PySAL ecosystem, the package provides a comprehensive framework for quantifying segregation across a wide range of demographic, socioeconomic, and geographic contexts. It supports both traditional aspatial measures and advanced spatially explicit approaches, enabling researchers to investigate how population groups are distributed across space and how these patterns evolve over time.

Segregation analysis is a central topic in urban studies, geography, sociology, demography, and public policy. The `segregation` package implements a large collection of segregation indices, including single-group, multigroup, local, and spatial measures. These methods capture different dimensions of segregation such as evenness, exposure, concentration, clustering, centralization, and spatial proximity. The package accommodates a variety of spatial representations, including adjacency relationships, distance-based interactions, and network-based structures, allowing users to select measures that align with the theoretical and empirical characteristics of their study area.

Beyond point estimation, `segregation` provides a robust inferential framework for evaluating the statistical significance of segregation measures. Users can perform single-value and comparative hypothesis tests using simulation-based approaches, facilitating rigorous assessments of whether observed segregation patterns differ from those expected under specified null models. The package also includes decomposition methods that separate differences in segregation levels into components attributable to demographic composition and spatial structure, offering deeper insight into the processes underlying segregation dynamics.

Designed to operate natively with Pandas and GeoPandas data structures, `segregation` integrates seamlessly into contemporary Python-based spatial data science libraries and, therefore, minimizing friction in typical workflows. As a component of the Python Spatial Analysis Library (PySAL) ecosystem [@pysal2007; @rey2022pysalecosystem], it follows shared principles of interoperability, transparency, reproducibility, and methodological rigor. By providing accessible, well-documented, and extensible implementations of state-of-the-art segregation measures, `segregation` serves both applied researchers investigating social inequality and methodological developers advancing quantitative approaches to segregation analysis.


# Statement of need

Residential segregation, inequality, and spatial separation of population groups remain central concerns in urban studies, sociology, demography, geography, and public policy. Researchers and practitioners frequently seek to quantify the extent to which social groups are unevenly distributed across neighborhoods, cities, and regions, as well as to understand how segregation patterns vary across space and time. The increasing availability of demographic and socioeconomic data has expanded opportunities for segregation analysis, but it has also highlighted the need for accessible, transparent, and reproducible computational tools capable of implementing a wide range of segregation measures and inferential procedures.

Traditional software environments for segregation analysis often provide only a limited set of indices, rely on proprietary platforms, or require researchers to implement methods manually. These limitations can hinder reproducibility, reduce methodological transparency, and create barriers for comparing results across studies. Furthermore, advances in segregation research have introduced numerous spatially explicit measures, local indicators, decomposition techniques, and simulation-based inference methods that are not consistently available within existing analytical software.

Within the Python ecosystem, support for segregation analysis was historically fragmented. While libraries such as Pandas and GeoPandas provide essential data structures for handling tabular and spatial data, they do not natively implement segregation measures or associated inferential frameworks. Consequently, researchers often relied on custom scripts, isolated software implementations, or statistical packages in other programming languages, resulting in inconsistent workflows and difficulties in reproducing analyses.

`segregation` addresses these challenges by providing:

* A comprehensive and well-documented API for segregation measurement and analysis
* Native integration with Pandas, GeoPandas, and the broader PySAL ecosystem
* Support for a large collection of aspatial, spatial, local, and multigroup segregation indices
* Functions for measuring and visualizing multiscalar segregation profiles
* Simulation-based inference procedures for evaluating the statistical significance of segregation measures
* Decomposition methods that facilitate the investigation of demographic and spatial sources of segregation
* A strong emphasis on transparency, reproducibility, and methodological extensibility

These capabilities make `segregation` particularly valuable for researchers and practitioners in fields such as urban studies, sociology, demography, geography, public policy, public health, and regional science, where understanding patterns of spatial inequality and social separation is a fundamental analytical objective.

# State of the field

`segregation` is a component of the PySAL ecosystem, which provides a comprehensive suite of tools for spatial analysis in Python. Within this ecosystem, the packages are divided in four types (libraries examples in parenthesis):

* `Lib`: Core spatial data structures, file IO. Construction and interactive editing of spatial weights matrices & graphs. Alpha shapes, spatial indices, and spatial-topological relationships (`libpysal`)
* `Explore`: Modules to conduct exploratory analysis of spatial and spatio-temporal data (`esda`, `giddy`, `inequality`, etc.)
* `Model`: Estimation of spatial relationships in data with a variety of linear, generalized-linear, generalized-additive, and nonlinear models (`spreg`, `spopt`, `tobler`, `spglm`, etc.)
* `Viz`: Visualize patterns in spatial data to detect clusters, outliers, and hot-spots (`mapclassify`, `splot`, etc.)

`segregation` is present in the *Explore* set of libraries and addresses the specific problem of residential segregation.

Compared to desktop GIS platforms, `segregation` offers several advantages:

* **Reproducibility**: Workflows can be scripted and version-controlled
* **Transparency**: Methods and assumptions are explicit and inspectable
* **Extensibility**: Users can modify or extend algorithms for research purposes
* **Integration**: Interpolation can be embedded within larger data science pipelines, including machine learning and statistical modeling

Segregation phenomenon is also possible to be assessed with other softwares like the R packages `OasisR` [@tivadar2019oasisr] or `seg` [@segrhong2011]. However, PySAL's `segregation` module distinguishes itself in several critical ways: supports over 40 distinct segregation indices out-of-the-box, it allows simultaneous computation of multiple indices across multi-scalar geographical frameworks with fewer software dependencies, it integrates spatial networks using topological relationships allowing users to measure social separation based on real-world street networks rather than assuming straight-line (Euclidean), it also has a set of function for evaluating statistical significance using simulations, and it also allows users to decompose segregation scores—identifying exactly how much segregation occurs due to spatial context.

In addition, `segregation` provides a native solution for Python users, aligning with the growing adoption of Python in geospatial and data science communities.

# Software design

`segregation` is designed with attention to both flexibility, usability, and inter-operability between its core functions. The library is organized between different sub parts such as `singlegroup`, `multigroup`, `local`, `inference`, `decomposition`, `batch`, `network`, and `dynamics`.[^1] Segregation measures are built using Python classes which can be integrated in subsequent steps, such as inference or decomposition. The module is structured toward two kinds of segregation indices: 'spatially-explicit' and 'spatially-implicit'. The former includes space as part of its original formula. The latter uses the @reardonsullivan2004 approach to state that any segregation index is a spatial index if you transform the data properly.

[^1]: @cortes2020open introduced `segregation` but many new implementations were developed recently and the API suffered a major revision.

Additionally, `segregation` is developed with testing and documentation standards consistent with the Scientific Python ecosystem, ensuring reliability and maintainability.

<!-- "scikit-learn" like API? -->

## Core Functionality

`segregation` organizes its functionality around which type of segregation analysis is interested and each sub-part is explained as follows.


### Single and Multigroup Indices

Single group measures assess segregation between two different groups in a given location (i.e., one group vs. everyone else), Multi group segregation evaluates the simultaneous separation of all groups in a population (e.g., the distribution of White, Black, Asian, and Hispanic residents) across areas. 

Currently, `segregation` has over 40 indices available which represents, from our knowledge, the broader range of indices available for a user in any software. Also, the user can fit many indices at once with a wrapper function in the `batch` module.

### Local Indices

Unlike global indices that summarize an entire metropolitan area into a single value, local indices decompose segregation to the individual geographic unit level. Using these disaggregated measures helps identify precise spatial clusters where social isolation is most acute, uncovering micro-level dynamics that global metrics often mask. Currently, `segregation` has seven local indexes. 

### Multiscalar

The multiscalar profile [@reardon2008geographic] is a tool for measuring spatial segregation dynamics--the way that a segregation index changes values as the concept of a neighborhood changes, and what that tells us about macro versus micro patterns of segregation. The core idea is to calculate a segregation statistic, then expand the spatial scope of a neighborhood, recalculate the statistic, and repeat.

The package has a wrapper named `compute_multiscalar_profile` which can be used in a workflow to build these profiles.

### Simulation based Inference

PySAL's `segregation` module also addresses whether segregation index values are statistically significant under different specifications of a null hyphothesis using Monte Carlo simulations. Currently, for single value inference the module can generate approaches like generate bootstrap replications of the units with replacement (`bootstrap`), multinomial with restricted conditional probabilities (`systematic`), binomial with fixed parameters (`evenness`), geographic unit-level randomization (`geographic_permutation`), among others. For two-value inference, the user can specify resampling to generate distributions of the segregation index for each index (`bootstrap`), generate a synthetic dataset for each region through counterfactual estimates (`composition`), among others.

The correct specification of a null hyphothesis is a crucial part of this framework as different null hypotheses can lead to markedly different inferences, and also different segregation indexes, which can assess different segregation dimensions (i.e. evenness, exposure, concentration, centralization, and clustering) may not be appropriate for some specifications, specially for comparative inference.


### Decomposition

The decomposition approach implemented in the PySAL segregation module provides a framework for comparative segregation analysis that disentangles observed differences in segregation levels into two fundamental components: population composition and spatial structure. Building on the framework proposed by @rey2021comparative, the method uses spatially explicit counterfactual distributions combined with a Shapley value decomposition to determine how much of the difference between two segregation measures—whether across cities, time periods, or spatiotemporal contexts—is attributable to differences in demographic composition versus differences in the spatial arrangement of populations and areal units. 

The method can be applied to a variety of segregation indices and comparison settings, including cross-sectional, temporal, and spatiotemporal analyses. By moving beyond simple comparisons of index values, the decomposition provides a richer interpretation of segregation dynamics and helps identify whether differences arise primarily from changes in population composition or from the spatial organization of neighborhoods. In the module, this functionality is available in the `DecomposeSegregation` class of the `decomposition` sub part.

## Example workflow

A typical workflow in `segregation` can be implemented as follows:

```python
from segregation.singlegroup import Dissim

result = Dissim(
    df_or_gdf,
    group_pop_var='group_A',
    total_pop_var='total_population'
)
```

This snippet estimates the Dissimilariy index of a specific sub-population (`group with characteristic A`) of your dataframe (`pandas` or `geopandas` DataFrames) and can accessed through the `statistic` attribute.

When additional information about within-zone heterogeneity is available, dasymetric interpolation can be used to refine estimates. For example, population counts may be redistributed using a land cover raster to exclude uninhabited areas:

```python
from tobler.dasymetric import masked_area_interpolate

result = masked_area_interpolate(
    raster="raster_file_name.tif",
    source_df,
    target_df,
    pixel_values = [21,22,23,24],
    extensive_variables=["population"]
)
```

This approach assumes the user have a raster data of his own that can be read by rasterio^[A common example is the ones available at the [National Land Cover Database](https://www.mrlc.gov/national-land-cover-database-nlcd-2016).]. In this example, `tobler` allows a flexible approach where the user can pass which pixels are to be assumed inhabited through `pixel_values` resulting in a more realistic spatial distribution. Similarly, the user can execute a model-based approach using the `tobler.model.glm` function.

\autoref{fig:emp_male_maps} illustrates an example comparing interpolated values derived from different spatial configurations, highlighting how results may vary depending on the underlying geometry and interpolation approach.

![Example of `tobler` usage for an extensive variable (male employment population) in Charleston, SC, comparing census tracts and ZCTAs.\label{fig:emp_male_maps}](figs/emp_male_maps.png)

# Research impact statement

The package is actively used by the research community to transfer the data between various types of geographic boundaries. This is not limited to specific applications but covers use cases from continental analysis of emissions and health [@laporta2024Urban], analysis of urban form and function [@fleischmann2022Geographical], redistribution of census data to school districts for assessment of the Clean School Bus Rebate Program [@osia2025Infrastructure], quantification of radon exposure [@lee2026Quantifyinga], or harmonization of vector and raster data for computer vision tasks [@fleischmann2024Decoding].

Moreover, the package is relied on in downstream software as `atlasbr` for harmonization of Brazilian urban data [@oliveira_paiva_neto_atlasbr], and is referred to in the `pygridmap` package by Eurostat [@grazzini_gaffuri_pygridmap] as a reference implementation.
The `tobler` package has made tangible contributions to spatial science, pedagogy, and applications in government and industry. In academia, the package is used as part of a data-processing pipeline for research that examines the spatial-contextual influence on a variety of outcomes, including segregation [@wei2022ReducingRacial], housing policy [@rey2022LegacyRedlining], education policy [@rey2024MeasuringSpatial; @osia2025InfrastructureEnvironmental], and pollution exposure [@lee2026QuantifyingMean; @laporta2024UrbanScaling]. It is also used in environmental science [@hu2023MethodologicalChallenges] and regionalization research [@feng2022MaxpcompactregionsProblem].

In spatial data science education, `tobler` has become an integral part of many many curricula. It is included in popular pedagogical resources including two textbooks [@reyGeographicDataScience2023; @knaapUrbanAnalysis2026], and is taught in graduate and undergraduate courses in univresities across the globe, including the University of California (Berkeley, Irvine, and Riverside campuses), San Diego State University, Charles University, University of Liverpool, Bristol University, the University of Chicago, Northern Arizona University, and Temple University.

**I took some liberty with a couple of these...we might want to check with Luc and Levi**

In the public sector, the `tobler` package is used as part of a processing pipeline that powers urban planning and policymaking, including two highly visible projects from the Turing Institute, [DemoLand](https://www.turing.ac.uk/research/research-projects/demoland) and [UrbanGrammar](https://www.turing.ac.uk/research/research-projects/urban-grammar). **Martin/Dani could you confirm and add a sentence or two?**?

# AI usage disclosure

No generative AI or LLMs were used for code production for `tobler` or the writing of this paper.

# Acknowledgements

`tobler` is developed as part of the PySAL community, which brings together researchers and developers working on spatial analysis methods and software. The project builds on decades of research in areal interpolation, dasymetric mapping, and spatial data science, and benefits from contributions across the open-source geospatial community.

The following acknowledgement applies to James D. Gaboardi:

> This manuscript has been authored in part by UT-Battelle LLC under contract DE-AC05-00OR22725 with the US Department of Energy (DOE). The US government retains and the publisher, by accepting the article for publication, acknowledges that the US government retains a nonexclusive, paid-up, irrevocable worldwide license to publish or reproduce the published form of this manuscript, or allow others to do so, for US government purposes. DOE will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).

# References
