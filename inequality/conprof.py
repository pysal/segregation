"""
Concentration Profile based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pandas as pd

__all__ = ['ConProf']


def _conprof(data, group_pop_var, total_pop_var, m = 1000):
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

    Attributes
    ----------

    r : float
        Concentration Profile Index

    Notes
    -----
    Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.

    """
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
    
    if any(data.total_pop_var < data.group_pop_var):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
   
    def calculate_vt(t):
        g_t_i = np.where(data.group_pop_var / data.total_pop_var >= t, 1, 0)
        v_t = (g_t_i * data.group_pop_var).sum() / data.group_pop_var.sum()
        return v_t
    
    grid = np.linspace(0, 1, m)
    curve = np.array(list(map(calculate_vt, grid)))
    
    threshold = data.group_pop_var.sum() / data.total_pop_var.sum()
    R = ((threshold - ((curve[grid < threshold]).sum() / m - (curve[grid >= threshold]).sum()/ m)) / (1 - threshold))
    
    return R


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

    Attributes
    ----------

    r : float
        Concentration Profile Index
        
    Examples
    --------
    In this example, we will calculate the concentration profile (R) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(url, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','nhblk10']]
    
    The value is estimated below.
    
    >>> conprof_index = ConProf(df, 'nhblk10', 'pop10')
    >>> conprof_index.r
    0.06393365660089256
        
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """

    def __init__(self, data, group_pop_var, total_pop_var, m = 1000):

        self.r = _conprof(data, group_pop_var, total_pop_var, m)


