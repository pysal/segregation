__version__ = "1.0.0"

"""
:mod:`segregation` --- Spatial/Non-Spatial Segregation Analysis
=================================================

"""

#from segregation import *

from segregation.dissimilarity import Dissim
from segregation.spatial_dissimilarity import Spatial_Dissim
from segregation.entropy import Entropy
from segregation.perimeter_area_ratio_spatial_dissimilarity import Perimeter_Area_Ratio_Spatial_Dissim
from segregation.absolute_centralization import Absolute_Centralization
from segregation.absolute_concentration import Absolute_Concentration
from segregation.atkinson import Atkinson
from segregation.bias_corrected_dissimilarity import Bias_Corrected_Dissim
from segregation.boundary_spatial_dissimilarity import Boundary_Spatial_Dissim
from segregation.conprof import Con_Prof
from segregation.correlationr import Correlation_R
from segregation.delta import Delta
from segregation.density_corrected_dissimilarity import Density_Corrected_Dissim
from segregation.exposure import Exposure
from segregation.gini_seg import Gini_Seg
from segregation.isolation import Isolation
from segregation.modified_dissimilarity import Modified_Dissim
from segregation.modified_gini_seg import Modified_Gini_Seg
from segregation.relative_centralization import Relative_Centralization
from segregation.relative_clustering import Relative_Clustering
from segregation.relative_concentration import Relative_Concentration
from segregation.spatial_exposure import Spatial_Exposure
from segregation.spatial_isolation import Spatial_Isolation
from segregation.spatial_prox_profile import Spatial_Prox_Prof
from segregation.spatial_proximity import Spatial_Proximity

from segregation.infer_segregation import Infer_Segregation
from segregation.compare_segregation import Compare_Segregation

from segregation.profile_wrappers import *