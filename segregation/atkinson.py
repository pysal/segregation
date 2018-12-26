"""
Atkinson Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pandas as pd

__all__ = ['Atkinson']


def _atkinson(data, group_pop_var, total_pop_var, b = 0.5):
    """
    Calculation of Entropy index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    b             : float
                    The shape parameter, between 0 and 1, that determines how to weight the increments to segregation contributed by different portions of the Lorenz curve.

    Attributes
    ----------

    statistic : float
                Atkinson Index
    
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """
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
    
    core_data = data[['group_pop_var', 'total_pop_var']]
    
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
    
    Firstly, we need to read the data:
    
    >>> This example uses all census data that the user must provide your own copy of the external database.
    >>> A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/osnap/tree/master/osnap/data.
    >>> After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','nhblk10']]
    
    The value is estimated below.
    
    >>> atkinson_index = Atkinson(df, 'nhblk10', 'pop10', b = 0.5)
    >>> atkinson_index.statistic
    0.16722406110274002
       
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """

    def __init__(self, data, group_pop_var, total_pop_var, b = 0.5):
        
        aux = _atkinson(data, group_pop_var, total_pop_var, b)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _atkinson

