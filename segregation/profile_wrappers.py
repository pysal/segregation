"""
Profile Wrappers for Segregation measures
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"


from segregation.dissimilarity import Dissim
from segregation.entropy import Entropy
from segregation.atkinson import Atkinson
from segregation.bias_corrected_dissimilarity import Bias_Corrected_Dissim
from segregation.conprof import Con_Prof
from segregation.correlationr import Correlation_R
from segregation.density_corrected_dissimilarity import Density_Corrected_Dissim
from segregation.exposure import Exposure
from segregation.gini_seg import Gini_Seg
from segregation.isolation import Isolation
from segregation.modified_dissimilarity import Modified_Dissim
from segregation.modified_gini_seg import Modified_Gini_Seg

from segregation.spatial_dissimilarity import Spatial_Dissim
from segregation.perimeter_area_ratio_spatial_dissimilarity import Perimeter_Area_Ratio_Spatial_Dissim
from segregation.absolute_centralization import Absolute_Centralization
from segregation.absolute_concentration import Absolute_Concentration
from segregation.boundary_spatial_dissimilarity import Boundary_Spatial_Dissim
from segregation.delta import Delta
from segregation.relative_centralization import Relative_Centralization
from segregation.relative_clustering import Relative_Clustering
from segregation.relative_concentration import Relative_Concentration
from segregation.spatial_exposure import Spatial_Exposure
from segregation.spatial_isolation import Spatial_Isolation
from segregation.spatial_prox_profile import Spatial_Prox_Prof
from segregation.spatial_proximity import Spatial_Proximity

__all__ = ['Profile_Non_Spatial_Segregation', 'Profile_Spatial_Segregation', 'Profile_Segregation']

def _profile_non_spatial_segregation(data, group_pop_var, total_pop_var):
    '''
    Perform point estimation of selected non spatial segregation measures at once

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    Attributes
    ----------

    profile     : dict
                  A dictionary containing the name of the measure and the point estimation.
    
    '''
    
    D = Dissim(data, group_pop_var, total_pop_var)
    G = Gini_Seg(data, group_pop_var, total_pop_var)
    H = Entropy(data, group_pop_var, total_pop_var)
    A = Atkinson(data, group_pop_var, total_pop_var)
    xPy = Exposure(data, group_pop_var, total_pop_var)
    xPx = Isolation(data, group_pop_var, total_pop_var)
    R = Con_Prof(data, group_pop_var, total_pop_var)
    Dbc = Bias_Corrected_Dissim(data, group_pop_var, total_pop_var)
    Ddc = Density_Corrected_Dissim(data, group_pop_var, total_pop_var)
    V = Correlation_R(data, group_pop_var, total_pop_var)
    Dct = Modified_Dissim(data, group_pop_var, total_pop_var)
    Gct = Modified_Gini_Seg(data, group_pop_var, total_pop_var)
    
    dictionary = {'Dissimilarity': D.statistic,
                  'Gini': G.statistic,
                  'Entropy': H.statistic,
                  'Atkinson': A.statistic,
                  'Exposure': xPy.statistic,
                  'Isolation': xPx.statistic,
                  'Concentration Profile': R.statistic,
                  'Bias Corrected Dissimilarity': Dbc.statistic,
                  'Density Corrected Dissimilarity': Ddc.statistic,
                  'Correlation Ratio': V.statistic,
                  'Modified Dissimilarity': Dct.statistic,
                  'Modified Gini': Gct.statistic
                  }
    
    return dictionary



class Profile_Non_Spatial_Segregation:
    '''
    Perform point estimation of selected non spatial segregation measures at once

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    Attributes
    ----------

    profile     : dict
                  A dictionary containing the name of the measure and the point estimation.
    
    '''

    def __init__(self, data, group_pop_var, total_pop_var):
        
        aux = _profile_non_spatial_segregation(data, group_pop_var, total_pop_var)

        self.profile = aux
        
        
        








def _profile_spatial_segregation(data, group_pop_var, total_pop_var):
    '''
    Perform point estimation of selected spatial segregation measures at once

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    Attributes
    ----------

    profile     : dict
                  A dictionary containing the name of the measure and the point estimation.
    
    '''
    
    SD = Spatial_Dissim(data, group_pop_var, total_pop_var)
    PARD = Perimeter_Area_Ratio_Spatial_Dissim(data, group_pop_var, total_pop_var)
    BSD = Boundary_Spatial_Dissim(data, group_pop_var, total_pop_var)
    ACE = Absolute_Centralization(data, group_pop_var, total_pop_var)
    ACO = Absolute_Concentration(data, group_pop_var, total_pop_var)
    DEL = Delta(data, group_pop_var, total_pop_var)
    RCE = Relative_Centralization(data, group_pop_var, total_pop_var)
    RCL = Relative_Clustering(data, group_pop_var, total_pop_var)
    RCO = Relative_Concentration(data, group_pop_var, total_pop_var)
    SxPy = Spatial_Exposure(data, group_pop_var, total_pop_var)
    SxPx = Spatial_Isolation(data, group_pop_var, total_pop_var)
    SPP = Spatial_Prox_Prof(data, group_pop_var, total_pop_var)
    SP = Spatial_Proximity(data, group_pop_var, total_pop_var)
    
    dictionary = {'Spatial Dissimilarity': SD.statistic,
                  'Absolute Centralization': ACE.statistic,
                  'Absolute Concentration': ACO.statistic,
                  'Delta': DEL.statistic,
                  'Relative Centralization': RCE.statistic,
                  'Relative Clustering': RCL.statistic,
                  'Relative Concentration': RCO.statistic,
                  'Spatial Exposure': SxPy.statistic,
                  'Spatial Isolation': SxPx.statistic,
                  'Spatial Proximity Profile': SPP.statistic,
                  'Spatial Proximity': SP.statistic,
                  'Boundary Spatial Dissimilarity': BSD.statistic,
                  'Perimeter Area Ratio Spatial Dissimilarity': PARD.statistic
                  }
    
    return dictionary



class Profile_Spatial_Segregation:
    '''
    Perform point estimation of selected spatial segregation measures at once

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    Attributes
    ----------

    profile     : dict
                  A dictionary containing the name of the measure and the point estimation.
    
    '''

    def __init__(self, data, group_pop_var, total_pop_var):
        
        aux = _profile_spatial_segregation(data, group_pop_var, total_pop_var)

        self.profile = aux


















def _profile_segregation(data, group_pop_var, total_pop_var):
    '''
    Perform point estimation of selected segregation measures at once

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    Attributes
    ----------

    profile     : dict
                  A dictionary containing the name of the measure and the point estimation.
    
    '''
    
    x = Profile_Non_Spatial_Segregation(data, group_pop_var, total_pop_var).profile
    y = Profile_Spatial_Segregation(data, group_pop_var, total_pop_var).profile
    
    x.update(y)
    
    return x



class Profile_Segregation:
    '''
    Perform point estimation of selected segregation measures at once

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    Attributes
    ----------

    profile     : dict
                  A dictionary containing the name of the measure and the point estimation.
    
    '''

    def __init__(self, data, group_pop_var, total_pop_var):
        
        aux = _profile_segregation(data, group_pop_var, total_pop_var)

        self.profile = aux
        