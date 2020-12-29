.. _api_ref:

.. currentmodule:: segregation

API reference
=============

Aspatial Indices
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
   
      aspatial.Dissim 
      aspatial.GiniSeg
      aspatial.Entropy
      aspatial.Isolation
      aspatial.Exposure
      aspatial.Atkinson
      aspatial.CorrelationR
      aspatial.ConProf
      aspatial.ModifiedDissim
      aspatial.ModifiedGiniSeg
      aspatial.BiasCorrectedDissim
      aspatial.DensityCorrectedDissim
      aspatial.MinMax

Spatial Indices
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
   
      spatial.SpatialProxProf
      spatial.SpatialDissim
      spatial.BoundarySpatialDissim
      spatial.PerimeterAreaRatioSpatialDissim
      spatial.DistanceDecayIsolation
      spatial.DistanceDecayExposure
      spatial.SpatialProximity
      spatial.AbsoluteClustering
      spatial.RelativeClustering
      spatial.Delta
      spatial.AbsoluteConcentration
      spatial.RelativeConcentration
      spatial.AbsoluteCentralization
      spatial.RelativeCentralization
      spatial.SpatialMinMax

Multi-Scalar Spatial Measures
----------------------------------

.. currentmodule:: segregation

.. autosummary::
 :toctree: generated/

 	   spatial.compute_segregation_profile

Multigroup Indices
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
   
      aspatial.MultiDissim
      aspatial.MultiGiniSeg
      aspatial.MultiNormalizedExposure
      aspatial.MultiInformationTheory
      aspatial.MultiRelativeDiversity
      aspatial.MultiSquaredCoefficientVariation
      aspatial.MultiDiversity
      aspatial.SimpsonsConcentration
      aspatial.SimpsonsInteraction
      aspatial.MultiDivergence
	  
Local Indices
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
   
      local.MultiLocationQuotient
      local.MultiLocalDiversity
      local.MultiLocalEntropy
      local.MultiLocalSimpsonInteraction
      local.MultiLocalSimpsonConcentration
      local.LocalRelativeCentralization
	  
Batch Compute Wrappers
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
   
	  compute_all.ComputeAllAspatialSegregation
	  compute_all.ComputeAllSpatialSegregation
	  compute_all.ComputeAllSegregation
	  
Inference Wrappers
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
   
	  inference.SingleValueTest
	  inference.TwoValueTest
	  
Decomposition
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
  
      decomposition.DecomposeSegregation

Network
---------------------

.. currentmodule:: segregation

.. autosummary::
 :toctree: generated/

      network.get_osm_network
      network.calc_access

Util
----------------

.. currentmodule:: segregation

.. autosummary::
 :toctree: generated/

      util.project_gdf