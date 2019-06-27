.. _api_ref:

.. currentmodule:: segregation

API reference
=============

Aspatial Indices
---------------------
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

Spatial Indices
---------------------
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
	  
Multigroup Indices
---------------------
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
.. autosummary::
   :toctree: generated/
   
      local.MultiLocationQuotient
      local.MultiLocalDiversity
      local.MultiLocalEntropy
      local.MultiLocalSimpsonInteraction
      local.MultiLocalSimpsonConcentration
      local.LocalRelativeCentralization
	  
Profile Wrappers
---------------------
.. autosummary::
   :toctree: generated/
   
	  compute_all.ComputeAllAspatialSegregation
	  compute_all.ComputeAllSpatialSegregation
	  compute_all.ComputeAllSegregation
	  
Inference Wrappers
---------------------
.. autosummary::
   :toctree: generated/
   
	  inference.InferSegregation
	  inference.CompareSegregation
	  
Decomposition
---------------------
.. autosummary::
   :toctree: generated/
   
      decomposition.DecomposeSegregation
