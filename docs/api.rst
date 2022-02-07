.. _api_ref:

.. currentmodule:: segregation

API reference
=============

Single Group Indices
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
   
      singlegroup.AbsoluteCentralization
      singlegroup.AbsoluteClustering
      singlegroup.AbsoluteConcentration
      singlegroup.Atkinson
      singlegroup.BiasCorrectedDissim
      singlegroup.BoundarySpatialDissim
      singlegroup.ConProf
      singlegroup.CorrelationR
      singlegroup.Delta
      singlegroup.DensityCorrectedDissim
      singlegroup.Dissim 
      singlegroup.DistanceDecayInteraction
      singlegroup.DistanceDecayIsolation
      singlegroup.Entropy
      singlegroup.Gini
      singlegroup.Interaction
      singlegroup.Isolation
      singlegroup.MinMax
      singlegroup.ModifiedDissim
      singlegroup.ModifiedGini
      singlegroup.PARDissim
      singlegroup.RelativeCentralization
      singlegroup.RelativeClustering
      singlegroup.RelativeConcentration
      singlegroup.SpatialDissim
      singlegroup.SpatialProximity
      singlegroup.SpatialProxProf

Multigroup Indices
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
   
      multigroup.MultiDissim
      multigroup.MultiDivergence
      multigroup.MultiDiversity      
      multigroup.MultiGini
      multigroup.MultiInfoTheory
      multigroup.MultiNormExposure
      multigroup.MultiRelativeDiversity
      multigroup.MultiSquaredCoefVar
      multigroup.SimpsonsConcentration
      multigroup.SimpsonsInteraction
	  
Local Indices
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/

      local.LocalKLDivergence
      local.LocalRelativeCentralization
      local.MultiLocalDiversity
      local.MultiLocalEntropy
      local.MultiLocationQuotient
      local.MultiLocalSimpsonInteraction
      local.MultiLocalSimpsonConcentration
	  
Dynamics
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
   
      dynamics.compute_multiscalar_profile
  
Batch Computation
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
   
	  batch.batch_compute_singlegroup
	  batch.batch_compute_multigroup
	  batch.batch_multiscalar_singlegroup
	  batch.batch_multiscalar_multigroup
	  
Inference 
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
   
	  inference.SingleValueTest
	  inference.TwoValueTest

     inference.simulate_bootstrap_resample
     inference.sim_composition
     inference.sim_dual_composition
     inference.simulate_evenness
     inference.simulate_evenness_geo_permutation
     inference.simulate_geo_permutation
     inference.simulate_null
     inference.simulate_person_permutation
     inference.sim_share
     inference.simulate_systematic_randomization
     inference.simulate_systematic_geo_permutation

	  
Decomposition
---------------------

.. currentmodule:: segregation

.. autosummary::
   :toctree: generated/
  
      decomposition.DecomposeSegregation


Util
----------------

.. currentmodule:: segregation

.. autosummary::
 :toctree: generated/

      util.get_osm_network