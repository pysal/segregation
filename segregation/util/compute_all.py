"""
Compute Several Segregation measures at once
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import pandas as pd
from segregation.aspatial import *
from segregation.spatial import *

__all__ = [
    'Compute_All_Aspatial_Segregation', 'Compute_All_Spatial_Segregation',
    'Compute_All_Segregation'
]


def _compute_all_aspatial_segregation(data, group_pop_var, total_pop_var):
    '''
    Perform point estimation of selected aspatial segregation measures at once

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    Returns
    -------

    computed      : a pandas DataFrame containing the name of the measure and the point estimation.
    
    Notes
    -----
    Currently, works with the default input parameters of the functions.
    
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

    dictionary = {
        'Dissimilarity': D.statistic,
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
    
    d = {'Measure': list(dictionary.keys()), 'Value': list(dictionary.values())}

    computed = pd.DataFrame(data = d)

    return computed


class Compute_All_Aspatial_Segregation:
    '''
    Perform point estimation of selected Aspatial segregation measures at once

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    Attributes
    ----------

    computed      : a pandas DataFrame containing the name of the measure and the point estimation.
    
    Examples
    --------
    The Compute_All comprises simple and quick functions to assess multiple segregation measures at once in a dataset. It uses all the default parameters and returns an object that has an attribute (.computed) of a dictionary with summary of all values fitted.

    Firstly, we need to import the libraries and functions to be used.
    
    >>> import geopandas as gpd
    >>> import segregation
    >>> import libpysal
    >>> from segregation.util import Compute_All_Aspatial_Segregation
    
    Then it's time to load some data to estimate segregation. We use the data of 2000 Census Tract Data for the metropolitan area of Sacramento, CA, USA.

    We use a geopandas dataframe available in PySAL examples repository.

    For more information about the data: https://github.com/pysal/libpysal/tree/master/libpysal/examples/sacramento2
    
    >>> s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    
    The data have several demographic variables. We are going to assess the segregation of the Hispanic Population (variable 'HISP_'). For this, we only extract some columns of the geopandas dataframe.
    
    >>> gdf = s_map[['geometry', 'HISP_', 'TOT_POP']]
    
    Now the measures are fitted.
    
    >>> aspatial_fit = Compute_All_Aspatial_Segregation(gdf, 'HISP_', 'TOT_POP')
    >>> aspatial_fit.computed
    
    '''

    def __init__(self, data, group_pop_var, total_pop_var):

        aux = _compute_all_aspatial_segregation(data, group_pop_var,
                                            total_pop_var)

        self.computed = aux


def _compute_all_spatial_segregation(data, group_pop_var, total_pop_var):
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

    computed      : a pandas DataFrame containing the name of the measure and the point estimation.
                  
    '''

    SD = Spatial_Dissim(data, group_pop_var, total_pop_var)
    PARD = Perimeter_Area_Ratio_Spatial_Dissim(data, group_pop_var,
                                               total_pop_var)
    BSD = Boundary_Spatial_Dissim(data, group_pop_var, total_pop_var)
    ACE = Absolute_Centralization(data, group_pop_var, total_pop_var)
    ACO = Absolute_Concentration(data, group_pop_var, total_pop_var)
    DEL = Delta(data, group_pop_var, total_pop_var)
    RCE = Relative_Centralization(data, group_pop_var, total_pop_var)
    ACL = Absolute_Clustering(data, group_pop_var, total_pop_var)
    RCL = Relative_Clustering(data, group_pop_var, total_pop_var)
    RCO = Relative_Concentration(data, group_pop_var, total_pop_var)
    DDxPy = Distance_Decay_Exposure(data, group_pop_var, total_pop_var)
    DDxPx = Distance_Decay_Isolation(data, group_pop_var, total_pop_var)
    SPP = Spatial_Prox_Prof(data, group_pop_var, total_pop_var)
    SP = Spatial_Proximity(data, group_pop_var, total_pop_var)

    dictionary = {
        'Spatial Dissimilarity': SD.statistic,
        'Absolute Centralization': ACE.statistic,
        'Absolute Clustering': ACL.statistic,
        'Absolute Concentration': ACO.statistic,
        'Delta': DEL.statistic,
        'Relative Centralization': RCE.statistic,
        'Relative Clustering': RCL.statistic,
        'Relative Concentration': RCO.statistic,
        'Distance Decay Exposure': DDxPy.statistic,
        'Distance Decay Isolation': DDxPx.statistic,
        'Spatial Proximity Profile': SPP.statistic,
        'Spatial Proximity': SP.statistic,
        'Boundary Spatial Dissimilarity': BSD.statistic,
        'Perimeter Area Ratio Spatial Dissimilarity': PARD.statistic
    }

    d = {'Measure': list(dictionary.keys()), 'Value': list(dictionary.values())}

    computed = pd.DataFrame(data = d)

    return computed


class Compute_All_Spatial_Segregation:
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

    computed      : a pandas DataFrame containing the name of the measure and the point estimation.
    
    Examples
    --------
    The Compute_All comprises simple and quick functions to assess multiple segregation measures at once in a dataset. It uses all the default parameters and returns an object that has an attribute (.computed) of a dictionary with summary of all values fitted.

    Firstly, we need to import the libraries and functions to be used.
    
    >>> import geopandas as gpd
    >>> import segregation
    >>> import libpysal
    >>> from segregation.util import Compute_All_Spatial_Segregation
    
    Then it's time to load some data to estimate segregation. We use the data of 2000 Census Tract Data for the metropolitan area of Sacramento, CA, USA.

    We use a geopandas dataframe available in PySAL examples repository.

    For more information about the data: https://github.com/pysal/libpysal/tree/master/libpysal/examples/sacramento2
    
    >>> s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    
    The data have several demographic variables. We are going to assess the segregation of the Hispanic Population (variable 'HISP_'). For this, we only extract some columns of the geopandas dataframe.
    
    >>> gdf = s_map[['geometry', 'HISP_', 'TOT_POP']]
    
    Now the measures are fitted.
    
    >>> spatial_fit = Compute_All_Spatial_Segregation(gdf, 'HISP_', 'TOT_POP')
    >>> spatial_fit.computed
    
    '''

    def __init__(self, data, group_pop_var, total_pop_var):

        aux = _compute_all_spatial_segregation(data, group_pop_var, total_pop_var)

        self.computed = aux


def _compute_all_segregation(data, group_pop_var, total_pop_var):
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

    computed      : a pandas DataFrame containing the name of the measure and the point estimation.
    
    '''

    x = Compute_All_Aspatial_Segregation(data, group_pop_var,
                                     total_pop_var).computed
    y = Compute_All_Spatial_Segregation(data, group_pop_var, total_pop_var).computed

    z = pd.concat([x, y], ignore_index = True)

    return z


class Compute_All_Segregation:
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

    computed      : a pandas DataFrame containing the name of the measure and the point estimation.
    
    Examples
    --------
    The Compute_All comprises simple and quick functions to assess multiple segregation measures at once in a dataset. It uses all the default parameters and returns an object that has an attribute (.computed) of a dictionary with summary of all values fitted.

    Firstly, we need to import the libraries and functions to be used.
    
    >>> import geopandas as gpd
    >>> import segregation
    >>> import libpysal
    >>> from segregation.util import Compute_All_Segregation
    
    Then it's time to load some data to estimate segregation. We use the data of 2000 Census Tract Data for the metropolitan area of Sacramento, CA, USA.

    We use a geopandas dataframe available in PySAL examples repository.

    For more information about the data: https://github.com/pysal/libpysal/tree/master/libpysal/examples/sacramento2
    
    >>> s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    
    The data have several demographic variables. We are going to assess the segregation of the Hispanic Population (variable 'HISP_'). For this, we only extract some columns of the geopandas dataframe.
    
    >>> gdf = s_map[['geometry', 'HISP_', 'TOT_POP']]
    
    Now the measures are fitted.
    
    >>> segregation_fit = Compute_All_Segregation(gdf, 'HISP_', 'TOT_POP')
    >>> segregation_fit.computed
    
    '''

    def __init__(self, data, group_pop_var, total_pop_var):

        aux = _compute_all_segregation(data, group_pop_var, total_pop_var)

        self.computed = aux
