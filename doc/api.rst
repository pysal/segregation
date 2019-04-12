.. _api_ref:

.. currentmodule:: segregation

API reference
=============

A-spatial Indices
---------------------
.. autosummary::
   :toctree: generated/
   
      non_spatial_indexes.Dissim 
      non_spatial_indexes.Gini_Seg
      non_spatial_indexes.Entropy
      non_spatial_indexes.Isolation
      non_spatial_indexes.Exposure
      non_spatial_indexes.Atkinson
      non_spatial_indexes.Correlation_R
      non_spatial_indexes.Con_Prof
      non_spatial_indexes.Modified_Dissim
      non_spatial_indexes.Modified_Gini_Seg
      non_spatial_indexes.Bias_Corrected_Dissim
      non_spatial_indexes.Density_Corrected_Dissim

Spatial Indices
---------------------
.. autosummary::
   :toctree: generated/
   
      spatial_indexes.Spatial_Prox_Prof
      spatial_indexes.Spatial_Dissim
      spatial_indexes.Boundary_Spatial_Dissim
      spatial_indexes.Perimeter_Area_Ratio_Spatial_Dissim
      spatial_indexes.Spatial_Isolation
      spatial_indexes.Spatial_Exposure
      spatial_indexes.Spatial_Proximity
      spatial_indexes.Relative_Clustering
      spatial_indexes.Delta
      spatial_indexes.Absolute_Concentration
      spatial_indexes.Relative_Concentration
      spatial_indexes.Absolute_Centralization
      spatial_indexes.Relative_Centralization
      spatial_indexes.Spatial_Information_Theory
	  
Profile Wrappers
---------------------
.. autosummary::
   :toctree: generated/
   
      profile_wrappers.Profile_Non_Spatial_Segregation
	  profile_wrappers.Profile_Spatial_Segregation
	  profile_wrappers.Profile_Segregation
	  
Inference Wrappers
---------------------
.. autosummary::
   :toctree: generated/
   
      inference_wrappers.Infer_Segregation
	  inference_wrappers.Compare_Segregation
