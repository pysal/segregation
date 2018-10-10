"""
Exposure Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pandas as pd

__all__ = ['Exposure']


def _exposure(data, group_pop_var, total_pop_var):
    """
    Calculation of Exposure index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest (X)
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    xPy : float
          Exposure Index

    Notes
    -----
    The group of interest is labelled as group X, whilst Y is the complementary group. Groups X and Y are mutually excludent.
    
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    if any(data.total_pop_var < data.group_pop_var):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
   
    data = data.assign(xi = data.group_pop_var,
                       yi = data.total_pop_var - data.group_pop_var,
                       ti = data.total_pop_var)
    X = data.xi.sum()
    xPy = ((data.xi / X) * (data.yi / data.ti)).sum()
    
    return xPy


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

    Attributes
    ----------

    xPy : float
          Exposure Index
        
    Examples
    --------
    In this example, we will calculate the Exposure Index (xPy) for the Riverside County using the census tract data of 2010.
    The group of interest (X) is non-hispanic black people which is the variable nhblk10 in the dataset and the Y group is the other part of the population.
    
    Firstly, we need to read the data:
    
    >>> url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(url, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','nhblk10']]
    
    The value is estimated below.
    
    >>> exposure_index = exposure(df, 'nhblk10', 'pop10')
    >>> exposure_index.xPy
    0.886785172226587
    
    The interpretation of this number is that if you randomly pick a X member of a specific area, there is 88.68% of probability that this member shares a unit with a Y member.
    
    Notes
    -----
    The group of interest is labelled as group X, whilst Y is the complementary group. Groups X and Y are mutually excludent.
    
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """

    def __init__(self, data, group_pop_var, total_pop_var):

        self.xPy = _exposure(data, group_pop_var, total_pop_var)


