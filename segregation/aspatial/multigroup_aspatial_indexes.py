"""
Multigroup Aspatial based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"



import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances

__all__ = ['Multi_Dissim',
           'Multi_Gini_Seg',
           'Multi_Normalized_Exposure',
           'Multi_Information_Theory']

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
    
    core_c
    
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
        
        
        
def _multi_gini_seg(data, groups):
    """
    Calculation of Multigroup Gini index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    ----------

    statistic : float
                Multigroup Gini Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.

    """
    
    m = data.shape[1]
    
    df = np.array(data)
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    Pk = df.sum(axis = 0) / df.sum()
    Is = (Pk * (1 - Pk)).sum()
    
    elements_sum = np.empty(m)
    for k in range(m):
        aux = np.multiply(np.outer(ti, ti), manhattan_distances(pik[:,k].reshape(-1, 1))).sum()
        elements_sum[k] = aux
        
    multi_Gini_Seg = elements_sum.sum() / (2 * (T ** 2) * Is)
    
    return multi_Gini_Seg


class Multi_Gini_Seg:
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
    Available at multigroup_aspatial_example.ipynb

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_gini_seg(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_gini_seg
        


def _multi_normalized_exposure(data, groups):
    """
    Calculation of Multigroup Normalized Exposure index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    ----------

    statistic : float
                Multigroup Normalized Exposure Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.

    """
    
    df = np.array(data)
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    Pk = df.sum(axis = 0) / df.sum()
    
    MNE = ((ti[:,None] * (pik - Pk) ** 2) / (1 - Pk)).sum() / T
    
    return MNE


class Multi_Normalized_Exposure:
    """
    Calculation of Multigroup Normalized Exposure index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic : float
                Multigroup Normalized Exposure Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    Available at multigroup_aspatial_example.ipynb

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_normalized_exposure(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_normalized_exposure
        
        
        
def _multi_information_theory(data, groups):
    """
    Calculation of Multigroup Information Theory index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    ----------

    statistic : float
                Multigroup Information Theory Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.

    """
    
    df = np.array(data)
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    Pk = df.sum(axis = 0) / df.sum()
    
    # The natural logarithm is used, but this could be used with any base following Footnote 3 of pg. 37
    # of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    E = (Pk * np.log(1/Pk)).sum()
    
    MIT = np.nansum(ti[:,None] * pik * np.log(pik / Pk)) / (T*E)
    
    return MIT


class Multi_Information_Theory:
    """
    Calculation of Multigroup Information Theory index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic : float
                Multigroup Information Theory Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    Available at multigroup_aspatial_example.ipynb

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_information_theory(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_information_theory