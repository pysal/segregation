"""
Multigroup Aspatial based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"



import numpy as np
import pandas as pd

__all__ = ['Multi_Dissim']

def _multi_dissim(data, groups):
    """
    Calculation of Multigroup Dissimilarity index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    ----------

    statistic : float
                Multigroup Dissimilarity Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Sakoda, James M. "A generalized index of dissimilarity." Demography 18.2 (1981): 245-250.

    """
    
    df = np.array(data)
    
    n = df.shape[0]
    k = df.shape[1]
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    Pk = df.sum(axis = 0) / df.sum()
    
    Is = (Pk * (1 - Pk)).sum()
    
    multi_D = 1/(2 * T * Is) * np.multiply(abs(pik - Pk), np.repeat(ti, k, axis=0).reshape(n,k)).sum()
    
    return multi_D


class Multi_Dissim:
    """
    Calculation of Multigroup Dissimilarity index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic : float
                Multigroup Dissimilarity Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from segregation.multigroup_aspatial import Multi_Dissim
    
    Then, we read the data and extract only the necessary columns with an auxiliary list for fitting the index.
    
    >>> s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    >>> input_df = s_map[groups_list]
    
    The value is estimated below.
    
    >>> index = Multi_Dissim(input_df, groups_list)
    >>> index.statistic
    0.41340872573177806

    Notes
    -----
    Based on Sakoda, James M. "A generalized index of dissimilarity." Demography 18.2 (1981): 245-250.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_dissim(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_dissim