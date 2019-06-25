"""
Local based Segregation Metrics

Important: all classes that start with "Multi_" expects a specific type of input of multigroups since the index will be calculated using many groups.
On the other hand, other classes expects a single group for calculation of the metrics.
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Elijah Knaap <elijah.knaap@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import libpysal as lps

from segregation.spatial import Relative_Centralization


__all__ = [
    'Multi_Location_Quocient',
    'Multi_Local_Diversity',
    'Multi_Local_Entropy',
    'Multi_Local_Simpson_Interaction',
    'Multi_Local_Simpson_Concentration',
    'Local_Relative_Centralization'
]

def _multi_location_quocient(data, groups):
    """
    Calculation of Location Quocient index for each group and unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings of length k.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n,k)
                 Location Quocient values for each group and unit.
                 Column k has the Location Quocient of position k in groups.
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Isard, Walter. Methods of regional analysis. Vol. 4. Рипол Классик, 1967.
    
    Reference: :cite:`isard1967methods`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    n = df.shape[0]
    K = df.shape[1]
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    Xk = df.sum(axis = 0)
    
    multi_LQ = (df / np.repeat(ti, K, axis = 0).reshape(n,K)) / (Xk / T)
    
    return multi_LQ, core_data


class Multi_Location_Quocient:
    """
    Calculation of Location Quocient index for each group and unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings of length k.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistics : np.array(n,k)
                 Location Quocient values for each group and unit.
                 Column k has the Location Quocient of position k in groups.
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Isard, Walter. Methods of regional analysis. Vol. 4. Рипол Классик, 1967.
    
    Reference: :cite:`isard1967methods`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_location_quocient(data, groups)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _multi_location_quocient
        
        
        

def _multi_local_diversity(data, groups):
    """
    Calculation of Local Diversity index for each group and unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings of length k.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n,k)
                 Local Diversity values for each group and unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Theil, Henry. Statistical decomposition analysis; with applications in the social and administrative sciences. No. 04; HA33, T4.. 1972.
    
    Reference: :cite:`theil1972statistical`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    
    multi_LD = - np.nansum(pik * np.log(pik), axis = 1)
    
    return multi_LD, core_data


class Multi_Local_Diversity:
    """
    Calculation of Local Diversity index for each group and unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings of length k.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistics : np.array(n,k)
                 Local Diversity values for each group and unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Theil, Henry. Statistical decomposition analysis; with applications in the social and administrative sciences. No. 04; HA33, T4.. 1972.
    
    Reference: :cite:`theil1972statistical`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_local_diversity(data, groups)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _multi_local_diversity
        
        
def _multi_local_entropy(data, groups):
    """
    Calculation of Local Entropy index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n)
                 Local Entropy values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Eq. 6 of pg. 139 (individual unit case) of Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    
    Reference: :cite:`reardon2004measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    K = df.shape[1]
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    
    multi_LE = - np.nansum((pik * np.log(pik)) / np.log(K), axis = 1)
    
    return multi_LE, core_data


class Multi_Local_Entropy:
    """
    Calculation of Local Entropy index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistics : np.array(n)
                 Local Entropy values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Eq. 6 of pg. 139 (individual unit case) of Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    
    Reference: :cite:`reardon2004measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_local_entropy(data, groups)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _multi_local_entropy
        


def _multi_local_simpson_interaction(data, groups):
    """
    Calculation of Local Simpson Interaction index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n)
                 Local Simpson Interaction values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on the local version of Equation 1 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Simpson's interaction index can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to not belong to the same group.

    Higher values means lesser segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`reardon2002measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    
    local_SI = np.nansum(pik * (1 - pik), axis = 1)
    
    return local_SI, core_data


class Multi_Local_Simpson_Interaction:
    """
    Calculation of Local Simpson Interaction index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistics : np.array(n)
                 Local Simpson Interaction values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on the local version of Equation 1 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Simpson's interaction index can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to not belong to the same group.

    Higher values means lesser segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`reardon2002measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_local_simpson_interaction(data, groups)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _multi_local_simpson_interaction
        
        
def _multi_local_simpson_concentration(data, groups):
    """
    Calculation of Local Simpson concentration index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n)
                 Local Simpson concentration values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on the local version of Equation 1 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Simpson's concentration index can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to belong to the same group.

    Higher values means lesser segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`reardon2002measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    
    local_SC = np.nansum(pik * pik, axis = 1)
    
    return local_SC, core_data


class Multi_Local_Simpson_Concentration:
    """
    Calculation of Local Simpson concentration index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistics : np.array(n)
                 Local Simpson concentration values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on the local version of Equation 1 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Simpson's concentration index can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to belong to the same group.

    Higher values means lesser segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`reardon2002measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_local_simpson_concentration(data, groups)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _multi_local_simpson_concentration
        
        
def _local_relative_centralization(data, group_pop_var, total_pop_var, k_neigh = 5):
    """
    Calculation of Local Relative Centralization index for each unit

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    k_neigh       : integer greater than 0. Default is 5.
                    Number of assumed neighbors for local context (it uses k-nearest algorithm method)
                    
    Returns
    -------

    statistics : np.array(n)
                 Local Relative Centralization values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Folch, David C., and Sergio J. Rey. "The centralization index: A measure of local spatial segregation." Papers in Regional Science 95.3 (2016): 555-576.
    
    Reference: :cite:`folch2016centralization`.
    """
    
    data = data.copy()
    
    core_data = data[[group_pop_var, total_pop_var, 'geometry']]

    c_lons = data.centroid.map(lambda p: p.x)
    c_lats = data.centroid.map(lambda p: p.y)
    
    points = list(zip(c_lons, c_lats))
    kd = lps.cg.kdtree.KDTree(np.array(points))
    wnnk = lps.weights.KNN(kd, k = k_neigh)
    
    local_RCEs = np.empty(len(data))
    
    for i in range(len(data)):
    
        x = list(wnnk.neighbors.values())[i]
        x.append(list(wnnk.neighbors.keys())[i]) # Append in the end the current unit i inside the local context

        local_data = data.iloc[x,:].copy()
        
        # The center is given by the last position (i.e. the current unit i)
        local_RCE = Relative_Centralization(local_data, group_pop_var, total_pop_var, center = len(local_data) - 1)
        
        local_RCEs[i] = local_RCE.statistic
        
    return local_RCEs, core_data


class Local_Relative_Centralization:
    """
    Calculation of Local Relative Centralization index for each unit

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    k_neigh       : integer greater than 0. Default is 5.
                    Number of assumed neighbors for local context (it uses k-nearest algorithm method)
                    
    Returns
    -------

    statistics : np.array(n)
                 Local Relative Centralization values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Folch, David C., and Sergio J. Rey. "The centralization index: A measure of local spatial segregation." Papers in Regional Science 95.3 (2016): 555-576.
    
    Reference: :cite:`folch2016centralization`.
    """
    
    def __init__(self, data, group_pop_var, total_pop_var, k_neigh = 5):
        
        aux = _local_relative_centralization(data, group_pop_var, total_pop_var, k_neigh)

        self.statistics = aux[0]
        self.core_data  = aux[1]
        self._function  = _local_relative_centralization