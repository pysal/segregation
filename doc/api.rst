.. _api_ref:

.. currentmodule:: segregation

API reference
=============

Aspatial Indices
---------------------
.. autosummary::
   :toctree: generated/
   
      aspatial.Dissim 
      aspatial.Gini_Seg
      aspatial.Entropy
      aspatial.Isolation
      aspatial.Exposure
      aspatial.Atkinson
      aspatial.Correlation_R
      aspatial.Con_Prof
      aspatial.Modified_Dissim
      aspatial.Modified_Gini_Seg
      aspatial.Bias_Corrected_Dissim
      aspatial.Density_Corrected_Dissim

Spatial Indices
---------------------
.. autosummary::
   :toctree: generated/
   
      spatial.Spatial_Prox_Prof
      spatial.Spatial_Dissim
      spatial.Boundary_Spatial_Dissim
      spatial.Perimeter_Area_Ratio_Spatial_Dissim
      spatial.Distance_Decay_Isolation
      spatial.Distance_Decay_Exposure
      spatial.Spatial_Proximity
      spatial.Absolute_Clustering
      spatial.Relative_Clustering
      spatial.Delta
      spatial.Absolute_Concentration
      spatial.Relative_Concentration
      spatial.Absolute_Centralization
      spatial.Relative_Centralization
	  
Multigroup Indices
---------------------
.. autosummary::
   :toctree: generated/
   
      aspatial.Multi_Dissim
      aspatial.Multi_Gini_Seg
      aspatial.Multi_Normalized_Exposure
      aspatial.Multi_Information_Theory
      aspatial.Multi_Relative_Diversity
      aspatial.Multi_Squared_Coefficient_of_Variation
      aspatial.Multi_Diversity
      aspatial.Simpsons_Concentration
      aspatial.Simpsons_Interaction
      aspatial.Multi_Divergence
	  
Profile Wrappers
---------------------
.. autosummary::
   :toctree: generated/
   
	  profile.Profile_Aspatial_Segregation
	  profile.Profile_Spatial_Segregation
	  profile.Profile_Segregation
	  
Inference Wrappers
---------------------
.. autosummary::
   :toctree: generated/
   
	  inference.Infer_Segregation
	  inference.Compare_Segregation
	  
Decomposition
---------------------
.. autosummary::
   :toctree: generated/
   
      decomposition.Decompose_Segregation
