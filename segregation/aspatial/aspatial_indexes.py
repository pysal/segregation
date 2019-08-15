"""
Aspatial based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
import pandas as pd
import warnings
import geopandas as gpd

from scipy.stats import norm
from scipy.optimize import minimize

from segregation.util.util import _dep_message, DeprecationHelper, _nan_handle

# Including old and new api in __all__ so users can use both

__all__ = ['Dissim', 
           
           'Gini_Seg',
           'GiniSeg',
           
           'Entropy', 
           'Isolation',
           'Exposure',
           'Atkinson',
           
           'Correlation_R',
           'CorrelationR',
           
           'Con_Prof',
           'ConProf',
           
           'Modified_Dissim',
           'ModifiedDissim',
           
           'Modified_Gini_Seg',
           'ModifiedGiniSeg',
           
           'Bias_Corrected_Dissim',
           'BiasCorrectedDissim',
           
           'Density_Corrected_Dissim',
           'DensityCorrectedDissim',
           
           'MinMax']

# The Deprecation calls of the classes are located in the end of this script #




def _dissim(data, group_pop_var, total_pop_var, fillna = False):
    """
    Calculation of Dissimilarity index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Returns
    ----------

    statistic : float
                Dissimilarity Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    Reference: :cite:`massey1988dimensions`.
    
    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
        
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
   
    T = t.sum()
    P = x.sum() / T
    
    # If a unit has zero population, the group of interest frequency is zero
    pi = np.where(t == 0, 0, x / t)
    
    D = (((t * abs(pi - P)))/ (2 * T * P * (1 - P))).sum()
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return D, core_data


class Dissim:
    """
    Classic Dissimilarity Index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Attributes
    ----------

    statistic : float
                Dissimilarity Index
        
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
        
    Examples
    --------
    In this example, we will calculate the degree of dissimilarity (D) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import Dissim
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> dissim_index = Dissim(df, 'tractid', 'pop10')
    >>> dissim_index.statistic
    0.31565682496226544
    
    The interpretation of this value is that 31.57% of the non-hispanic black population would have to move to reach eveness in the Riverside County.
        
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, fillna = False):
        
        aux = _dissim(data, group_pop_var, total_pop_var, fillna)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _dissim
        
        
        
def _gini_seg(data, group_pop_var, total_pop_var, fillna = False):
    """
    Calculation of Gini Segregation index
    
    Parameters
    ----------
    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.
                    
    Returns
    ----------
    statistic : float
                Gini Segregation Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
                
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.
    
    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    if any(data.total_pop_var < data.group_pop_var):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
   
    T = data.total_pop_var.sum()
    P = data.group_pop_var.sum() / T
    
    # If a unit has zero population, the group of interest frequency is zero
    data = data.assign(ti = data.total_pop_var,
                       pi = np.where(data.total_pop_var == 0, 0, data.group_pop_var/data.total_pop_var))
    
    num = (np.matmul(np.array(data.ti)[np.newaxis].T, np.array(data.ti)[np.newaxis]) * abs(np.array(data.pi)[np.newaxis].T - np.array(data.pi)[np.newaxis])).sum()
    den = (2 * T**2 * P * (1-P))
    G = num / den
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return G, core_data


class GiniSeg:
    """
    Classic Gini Segregation Index
    
    Parameters
    ----------
    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.
                    
    Attributes
    ----------
    statistic : float
                Gini Segregation Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
                
    Examples
    --------
    In this example, we will calculate the Gini Segregation Index (G) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import GiniSeg
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> gini_seg_index = GiniSeg(df, 'tractid', 'pop10')
    >>> gini_seg_index.statistic
    0.44620350030600087
       
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, fillna = False):
        
        aux = _gini_seg(data, group_pop_var, total_pop_var, fillna)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _gini_seg
        
        
        
def _entropy(data, group_pop_var, total_pop_var, fillna = False):
    """
    Calculation of Entropy index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Returns
    ----------

    statistic : float
                Entropy Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
   
    T = t.sum()
    P = x.sum() / T
    
    # If a unit has zero population, the group of interest frequency is zero
    pi = np.where(t == 0, 0, x / t)
    
    E = P * np.log(1 / P) + (1 - P) * np.log(1 / (1 - P))
    Ei = pi * np.log(1 / pi) + (1 - pi) * np.log(1 / (1 - pi))
    H = np.nansum(t * (E - Ei) / (E * T)) # If some pi is zero, numpy will treat as zero
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return H, core_data


class Entropy:
    """
    Classic Entropy Index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Attributes
    ----------

    statistic : float
                Entropy Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
        
    Examples
    --------
    In this example, we will calculate the Entropy (H) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import Entropy
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> entropy_index = Entropy(df, 'tractid', 'pop10')
    >>> entropy_index.statistic
    0.08636489348167173
       
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, fillna = False):
        
        aux = _entropy(data, group_pop_var, total_pop_var, fillna)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _entropy
        
        
        
        
def _isolation(data, group_pop_var, total_pop_var, fillna = False):
    """
    Calculation of Isolation index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest (X)
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Returns
    ----------

    statistic : float
                Isolation Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
                
    Notes
    -----
    The group of interest is labelled as group X.
    
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
   
    X = x.sum()
    xPx = np.nansum((x / X) * (x / t))
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return xPx, core_data


class Isolation:
    """
    Classic Isolation Index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest (X)
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Attributes
    ----------

    statistic : float
                Isolation Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.   
                
    Examples
    --------
    In this example, we will calculate the Isolation Index (xPx) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import Isolation
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> isolation_index = Isolation(df, 'tractid', 'pop10')
    >>> isolation_index.statistic
    0.11321482777341298
    
    The interpretation of this number is that if you randomly pick a X member of a specific area, there is 11.32% of probability that this member shares a unit with another X member.
    
    Notes
    -----
    The group of interest is labelled as group X.
    
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, fillna = False):
        
        aux = _isolation(data, group_pop_var, total_pop_var, fillna)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _isolation
        
        
        
def _exposure(data, group_pop_var, total_pop_var, fillna = False):
    """
    Calculation of Exposure index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest (X)
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Returns
    ----------

    statistic : float
                Exposure Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 

    Notes
    -----
    The group of interest is labelled as group X, whilst Y is the complementary group. Groups X and Y are mutually excludent.
    
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
   
    yi = t - x
    
    X = x.sum()
    xPy = np.nansum((x / X) * (yi / t))
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return xPy, core_data


class Exposure:
    """
    Classic Exposure Index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest (X)
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Attributes
    ----------

    statistic : float
                Exposure Index

    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
                
    Examples
    --------
    In this example, we will calculate the Exposure Index (xPy) for the Riverside County using the census tract data of 2010.
    The group of interest (X) is non-hispanic black people which is the variable nhblk10 in the dataset and the Y group is the other part of the population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import Exposure
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> exposure_index = Exposure(df, 'tractid', 'pop10')
    >>> exposure_index.statistic
    0.886785172226587
    
    The interpretation of this number is that if you randomly pick a X member of a specific area, there is 88.68% of probability that this member shares a unit with a Y member.
    
    Notes
    -----
    The group of interest is labelled as group X, whilst Y is the complementary group. Groups X and Y are mutually excludent.
    
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, fillna = False):
        
        aux = _exposure(data, group_pop_var, total_pop_var, fillna)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _exposure




def _atkinson(data, group_pop_var, total_pop_var, b = 0.5, fillna = False):
    """
    Calculation of Atkinson index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    b             : float
                    The shape parameter, between 0 and 1, that determines how to weight the increments to segregation contributed by different portions of the Lorenz curve.
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Returns
    ----------

    statistic : float
                Atkinson Index
    
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if (not isinstance(b, float)):
        raise ValueError('The parameter b must be a float.')
        
    if ((b < 0) or (b > 1)):
        raise ValueError('The parameter b must be between 0 and 1.')
        
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
   
    T = t.sum()
    P = x.sum() / T
    
    # If a unit has zero population, the group of interest frequency is zero
    pi = np.where(t == 0, 0, x / t)
    
    A = 1 - (P / (1-P)) * abs((((1 - pi) ** (1-b) * pi ** b * t) / (P * T)).sum()) ** (1 / (1 - b))
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return A, core_data


class Atkinson:
    """
    Classic Atkinson Index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    b             : float
                    The shape parameter, between 0 and 1, that determines how to weight the increments to segregation contributed by different portions of the Lorenz curve.
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Attributes
    ----------

    statistic : float
                Atkison Index
    
    core_data : a pandas DataFrame
            A pandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the Atkinson Index (A) with the shape parameter (b) equals to 0.5 for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import Atkinson
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> atkinson_index = Atkinson(df, 'tractid', 'pop10', b = 0.5)
    >>> atkinson_index.statistic
    0.16722406110274002
       
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, b = 0.5, fillna = False):
        
        aux = _atkinson(data, group_pop_var, total_pop_var, b, fillna)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _atkinson
        
        
        
def _correlationr(data, group_pop_var, total_pop_var, fillna = False):
    """
    Calculation of Correlation Ratio index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest (X)
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Returns
    ----------

    statistic : float
                Correlation Ratio Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 

    Notes
    -----
    
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')

    X = x.sum()
    T = t.sum()
    P = X / T
    
    xPx = np.nansum((x / X) * (x / t))

    V = (xPx - P) / (1 - P)
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return V, core_data


class CorrelationR:
    """
    Classic Correlation Ratio Index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest (X)
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Attributes
    ----------

    statistic : float
                Correlation Ratio Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
        
    Examples
    --------
    In this example, we will calculate the Correlation Ratio Index (V) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import CorrelationR
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> correlationr_index = CorrelationR(df, 'tractid', 'pop10')
    >>> correlationr_index.statistic
    0.048716810856363923
    
    Notes
    -----
    
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, fillna = False):
        
        aux = _correlationr(data, group_pop_var, total_pop_var, fillna)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _correlationr
        
        
def _conprof(data, group_pop_var, total_pop_var, m = 1000, fillna = False):
    """
    Calculation of Concentration Profile

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    m             : int
                    a numeric value indicating the number of thresholds to be used. Default value is 1000. 
                    A large value of m creates a smoother-looking graph and a more precise concentration profile value but slows down the calculation speed.
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Returns
    ----------

    statistic : float
                Concentration Profile Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 

    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.

    Reference: :cite:`hong2014measuring`.

    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if(type(m) is not int):
        raise TypeError('m must be a string.')
        
    if(m < 2):
        raise ValueError('m must be greater than 1.')
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
   
    def calculate_vt(th):
        g_t_i = np.where(x / t >= th, 1, 0)
        v_t = (g_t_i * x).sum() / x.sum()
        return v_t
    
    grid = np.linspace(0, 1, m)
    curve = np.array(list(map(calculate_vt, grid)))
    
    threshold = x.sum() / t.sum()
    R = ((threshold - ((curve[grid < threshold]).sum() / m - (curve[grid >= threshold]).sum()/ m)) / (1 - threshold))
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return R, grid, curve, core_data


class ConProf:
    """
    Concentration Profile Index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    m             : int
                    a numeric value indicating the number of thresholds to be used. 
                    A large value of m creates a smoother-looking graph and a more precise concentration profile value but slows down the calculation speed.
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Attributes
    ----------

    statistic : float
                Concentration Profile Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
        
    Examples
    --------
    In this example, we will calculate the concentration profile (R) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import ConProf
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> conprof_index = ConProf(df, 'tractid', 'pop10')
    >>> conprof_index.statistic
    0.06393365660089256
    
    You can plot the profile curve with the plot method.
    
    >>> conprof_index.plot()
        
    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.
    
    Reference: :cite:`hong2014measuring`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, m = 1000, fillna = False):
        
        aux = _conprof(data, group_pop_var, total_pop_var, m, fillna)

        self.statistic = aux[0]
        self.grid      = aux[1]
        self.curve     = aux[2]
        self.core_data = aux[3]
        self._function = _conprof

    def plot(self):
        """
        Plot the Concentration Profile
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn('This method relies on importing `matplotlib`')
        graph = plt.scatter(self.grid, self.curve, s = 0.1)
        return graph
    
    
    
    
def _modified_dissim(data, group_pop_var, total_pop_var, iterations = 500, fillna = False):
    """
    Calculation of Modified Dissimilarity index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    iterations    : int
                    The number of iterations the evaluate average classic dissimilarity under eveness. Default value is 500.
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Returns
    ----------

    statistic : float
                Modified Dissimilarity Index (Dissimilarity from Carrington and Troske (1997))
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
                
    Notes
    -----
    Based on Carrington, William J., and Kenneth R. Troske. "On measuring segregation in samples with small units." Journal of Business & Economic Statistics 15.4 (1997): 402-409.
    
    Reference: :cite:`carrington1997measuring`.

    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if(type(iterations) is not int):
        raise TypeError('iterations must be an integer')
        
    if(iterations < 2):
        raise TypeError('iterations must be greater than 1.')
   
    D = _dissim(data, group_pop_var, total_pop_var)[0]
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    # core_data has to be in the beggining of the call because assign methods will be used later
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    p_null = x.sum() / t.sum()
    
    Ds = np.empty(iterations)
    
    for i in np.array(range(iterations)):

        freq_sim = np.random.binomial(n = np.array([t.tolist()]), 
                                      p = np.array([[p_null] * data.shape[0]]), 
                                      size = (1, data.shape[0])).tolist()[0]
        data = data.assign(group_pop_var = freq_sim)
        aux = _dissim(data, 'group_pop_var', 'total_pop_var')[0]
        Ds[i] = aux
        
    D_star = Ds.mean()
    
    if (D >= D_star):
        Dct = (D - D_star)/(1 - D_star)
    else:
        Dct = (D - D_star)/D_star
    
    return Dct, core_data


class ModifiedDissim:
    """
    Calculation of Modified Dissimilarity index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    iterations    : int
                    The number of iterations the evaluate average classic dissimilarity under eveness. Default value is 500.
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Attributes
    ----------

    statistic : float
                Modified Dissimilarity Index (Dissimilarity from Carrington and Troske (1997))
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.    
                
    Examples
    --------
    In this example, we will calculate the Modified Dissimilarity Index (Dct) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import ModifiedDissim
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> np.random.seed(1234)
    >>> modified_dissim_index = ModifiedDissim(df, 'tractid', 'pop10')
    >>> modified_dissim_index.statistic
    0.30009504639081996
     
    Notes
    -----
    Based on Carrington, William J., and Kenneth R. Troske. "On measuring segregation in samples with small units." Journal of Business & Economic Statistics 15.4 (1997): 402-409.

    Reference: :cite:`carrington1997measuring`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, iterations = 500, fillna = False):
        
        aux = _modified_dissim(data, group_pop_var, total_pop_var, iterations, fillna)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _modified_dissim
        
        
def _modified_gini_seg(data, group_pop_var, total_pop_var, iterations = 500, fillna = False):
    """
    Calculation of Modified Gini Segregation index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    iterations    : int
                    The number of iterations the evaluate average classic gini segregation under eveness. Default value is 500.
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Returns
    ----------

    statistic : float
                Modified Gini Segregation Index (Gini from Carrington and Troske (1997))
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
                
    Notes
    -----
    Based on Carrington, William J., and Kenneth R. Troske. "On measuring segregation in samples with small units." Journal of Business & Economic Statistics 15.4 (1997): 402-409.
    
    Reference: :cite:`carrington1997measuring`.

    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if(type(iterations) is not int):
        raise TypeError('iterations must be an integer')
        
    if(iterations < 2):
        raise TypeError('iterations must be greater than 1.')
   
    G = _gini_seg(data, group_pop_var, total_pop_var)[0]
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    # core_data has to be in the beggining of the call because assign methods will be used later
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    p_null = x.sum() / t.sum()
    
    Gs = np.empty(iterations)
    
    for i in np.array(range(iterations)):

        freq_sim = np.random.binomial(n = np.array([t.tolist()]), 
                                      p = np.array([[p_null] * data.shape[0]]), 
                                      size = (1, data.shape[0])).tolist()[0]
        data = data.assign(group_pop_var = freq_sim)
        aux = _gini_seg(data, 'group_pop_var', 'total_pop_var')[0]
        Gs[i] = aux
        
    G_star = Gs.mean()
    
    if (G >= G_star):
        Gct = (G - G_star)/(1 - G_star)
    else:
        Gct = (G - G_star)/G_star

    return Gct, core_data


class ModifiedGiniSeg:
    """
    Calculation of Modified Gini Segregation index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    iterations    : int
                    The number of iterations the evaluate average classic gini segregation under eveness. Default value is 500.
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Attributes
    ----------

    statistic : float
                Modified Gini Segregation Index (Gini from Carrington and Troske (1997))
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.     
                
    Examples
    --------
    In this example, we will calculate the Modified Gini Segregation Index (Gct) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import ModifiedGiniSeg
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> np.random.seed(1234)
    >>> modified_gini_seg_index = ModifiedGiniSeg(df, 'tractid', 'pop10')
    >>> modified_gini_seg_index.statistic
    0.4280279611418648
     
    Notes
    -----
    Based on Carrington, William J., and Kenneth R. Troske. "On measuring segregation in samples with small units." Journal of Business & Economic Statistics 15.4 (1997): 402-409.
    
    Reference: :cite:`carrington1997measuring`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, iterations = 500, fillna = False):
        
        aux = _modified_gini_seg(data, group_pop_var, total_pop_var, iterations, fillna)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _modified_gini_seg
        
        
        
        
def _bias_corrected_dissim(data, group_pop_var, total_pop_var, B = 500, fillna = False):
    """
    Calculation of Bias Corrected Dissimilarity index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    B             : int
                    The number of iterations to calculate Dissimilarity simulating randomness with multinomial distributions. Default value is 500.
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Returns
    ----------

    statistic : float
                Dissimilarity with Bias-Correction (bias correction from Allen, Rebecca et al. (2015))
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 

    Notes
    -----
    Based on Allen, Rebecca, et al. "More reliable inference for the dissimilarity index of segregation." The econometrics journal 18.1 (2015): 40-66.
    
    Reference: :cite:`allen2015more`.

    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if(type(B) is not int):
        raise TypeError('B must be an integer')
        
    if(B < 2):
        raise TypeError('B must be greater than 1.')
   
    D = _dissim(data, group_pop_var, total_pop_var)[0]
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    other_group_pop = t - x
    
    # Group 0: minority group
    p0_i = x / x.sum()
    n0   = x.sum()
    sim0 = np.random.multinomial(n0, p0_i, size = B)
    
    # Group 1: complement group
    p1_i = other_group_pop / other_group_pop.sum()
    n1   = other_group_pop.sum()
    sim1 = np.random.multinomial(n1, p1_i, size = B)

    
    Dbcs = np.empty(B)
    for i in np.array(range(B)):
        data_aux = {'simul_group': sim0[i].tolist(), 'simul_tot': (sim0[i] + sim1[i]).tolist()}
        df_aux = pd.DataFrame.from_dict(data_aux)
        Dbcs[i] = _dissim(df_aux, 'simul_group', 'simul_tot')[0]
        
    Db = Dbcs.mean()
    
    Dbc = 2 * D - Db
    Dbc # It expected to be lower than D, because D is upwarded biased
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
        
    return Dbc, core_data


class BiasCorrectedDissim:
    """
    Calculation of Bias Corrected Dissimilarity index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    B             : int
                    The number of iterations to calculate Dissimilarity simulating randomness with multinomial distributions. Default value is 500.
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Attributes
    ----------

    statistic : float
                Dissimilarity with Bias-Correction (bias correction from Allen, Rebecca et al. (2015))
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
        
    Examples
    --------
    In this example, we will calculate the Dissimilarity with Bias Correction (Dbc) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import BiasCorrectedDissim
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> np.random.seed(1234)
    >>> bias_corrected_dissim_index = BiasCorrectedDissim(df, 'tractid', 'pop10')
    >>> bias_corrected_dissim_index.statistic
    0.31484636081876954
     
    Notes
    -----
    Based on Allen, Rebecca, et al. "More reliable inference for the dissimilarity index of segregation." The econometrics journal 18.1 (2015): 40-66.
    
    Reference: :cite:`allen2015more`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, B = 500, fillna = False):
        
        aux = _bias_corrected_dissim(data, group_pop_var, total_pop_var, B, fillna)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _bias_corrected_dissim
        
        
def _density_corrected_dissim(data, group_pop_var, total_pop_var, xtol = 1e-5, fillna = False):
    """
    Calculation of Density Corrected Dissimilarity index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    xtol          : float
                    The degree of tolerance in the optimization process of returning optimal theta_j
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.

    Returns
    ----------

    statistic : float
                Dissimilarity with Density-Correction (density correction from Allen, Rebecca et al. (2015))
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 

    Notes
    -----
    Based on Allen, Rebecca, et al. "More reliable inference for the dissimilarity index of segregation." The econometrics journal 18.1 (2015): 40-66.
    
    Reference: :cite:`allen2015more`.

    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
        
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    g = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    if any(t < g):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
    
    other_group_pop = t - g
    
    # Group 0: minority group
    p0_i = g / g.sum()
    n0 = g.sum()
    
    # Group 1: complement group
    p1_i = other_group_pop / other_group_pop.sum()
    n1 = other_group_pop.sum()
    
    sigma_hat_j = np.sqrt(((p1_i * (1 - p1_i)) / n1) + ((p0_i * (1 - p0_i)) / n0))
    theta_hat_j = abs(p1_i - p0_i) / sigma_hat_j
    
    # Constructing function that returns $n(\hat{\theta}_j)$
    def return_optimal_theta(theta_j):
        
        def fold_norm(x):
            y = (-1) * (norm.pdf(x - theta_j) + norm.pdf(x + theta_j))
            return y
        
        initial_guesses = np.array(0)
        res = minimize(fold_norm, 
                       initial_guesses, 
                       method='nelder-mead',
                       options = {'xtol': xtol})
        return res.final_simplex[0][1][0]
        
    optimal_thetas = pd.Series(data = theta_hat_j).apply(return_optimal_theta)

    Ddc = np.multiply(sigma_hat_j, optimal_thetas).sum() / 2
    
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return Ddc, core_data


class DensityCorrectedDissim:
    """
    Calculation of Density Corrected Dissimilarity index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    xtol          : float
                    The degree of tolerance in the optimization process of returning optimal theta_j
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero.
                                  
    Attributes
    ----------

    statistic : float
                Dissimilarity with Density-Correction (density correction from Allen, Rebecca et al. (2015))
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate. 
        
    Examples
    --------
    In this example, we will calculate the Dissimilarity with Density Correction (Ddc) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.aspatial import DensityCorrectedDissim
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','tractid']]
    
    The value is estimated below.
    
    >>> density_corrected_dissim_index = DensityCorrectedDissim(df, 'tractid', 'pop10')
    >>> density_corrected_dissim_index.statistic
    0.29350643204887517
     
    Notes
    -----
    Based on Allen, Rebecca, et al. "More reliable inference for the dissimilarity index of segregation." The econometrics journal 18.1 (2015): 40-66.
    
    Reference: :cite:`allen2015more`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, xtol = 1e-5):
        
        aux = _density_corrected_dissim(data, group_pop_var, total_pop_var, xtol)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _density_corrected_dissim
        




def _min_max(data, group_pop_var, total_pop_var, fillna = False):
    """
    Calculation of the Aspatial version of SpatialMinMax

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero. 

    Returns
    ----------

    statistic : float
                MinMax Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on O'Sullivan & Wong (2007). A Surface‐Based Approach to Measuring Spatial Segregation.
    Geographical Analysis 39 (2). https://doi.org/10.1111/j.1538-4632.2007.00699.x

    Reference: :cite:`osullivanwong2007surface`.
    
    We'd like to thank @AnttiHaerkoenen for this contribution!
    
    """
    
    data = _nan_handle(data[[group_pop_var, total_pop_var]], fillna)
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
        
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    if any(data.total_pop_var < data.group_pop_var):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
   
    data['group_2_pop_var'] = data['total_pop_var'] - data['group_pop_var']
    
    data['group_1_pop_var_norm'] = data['group_pop_var'] / data['group_pop_var'].sum()
    data['group_2_pop_var_norm'] = data['group_2_pop_var'] / data['group_2_pop_var'].sum()
    
    density_1 = data['group_1_pop_var_norm'].values
    density_2 = data['group_2_pop_var_norm'].values
    densities = np.vstack([
        density_1,
        density_2
    ])
    v_union = densities.max(axis=0).sum()
    v_intersect = densities.min(axis=0).sum()
    
    MM = 1 - v_intersect / v_union
    
    if not isinstance(data, gpd.GeoDataFrame):
        core_data = data[['group_pop_var', 'total_pop_var']]
    
    else:    
        core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return MM, core_data


class MinMax:
    """
    Calculation of the Aspatial version of SpatialMinMax

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    fillna        : boolean. 
                    If `True`, will replace the NA's values to zero. 

    Attributes
    ----------

    statistic : float
                MinMax Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on O'Sullivan & Wong (2007). A Surface‐Based Approach to Measuring Spatial Segregation.
    Geographical Analysis 39 (2). https://doi.org/10.1111/j.1538-4632.2007.00699.x

    Reference: :cite:`osullivanwong2007surface`.
    
    We'd like to thank @AnttiHaerkoenen for this contribution!
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, fillna = False):
        
        aux = _min_max(data, group_pop_var, total_pop_var, fillna)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _min_max







# Deprecation Calls
        
msg = _dep_message("Gini_Seg", "GiniSeg")
Gini_Seg = DeprecationHelper(GiniSeg, message=msg)

msg = _dep_message("Correlation_R", "CorrelationR")
Correlation_R = DeprecationHelper(CorrelationR, message=msg)

msg = _dep_message("Con_Prof", "ConProf")
Con_Prof = DeprecationHelper(ConProf, message=msg)

msg = _dep_message("Modified_Dissim", "ModifiedDissim")
Modified_Dissim = DeprecationHelper(ModifiedDissim, message=msg)

msg = _dep_message("Modified_Gini_Seg", "ModifiedGiniSeg")
Modified_Gini_Seg = DeprecationHelper(ModifiedGiniSeg, message=msg)

msg = _dep_message("Bias_Corrected_Dissim", "BiasCorrectedDissim")
Bias_Corrected_Dissim = DeprecationHelper(BiasCorrectedDissim, message=msg)

msg = _dep_message("Density_Corrected_Dissim", "DensityCorrectedDissim")
Density_Corrected_Dissim = DeprecationHelper(DensityCorrectedDissim, message=msg)