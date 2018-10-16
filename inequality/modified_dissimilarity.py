"""
Modified Dissimilarity based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pandas as pd
from dissimilarity import _dissim

__all__ = ['Modified_Dissim']


def _modified_dissim(data, group_pop_var, total_pop_var, iterations = 500):
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

    Attributes
    ----------

    statistic : float
                Modified Dissimilarity Index (Dissimilarity from Carrington and Troske (1997))

    Notes
    -----
    Based on Carrington, William J., and Kenneth R. Troske. "On measuring segregation in samples with small units." Journal of Business & Economic Statistics 15.4 (1997): 402-409.

    """
    if(type(iterations) is not int):
        raise TypeError('iterations must be an integer')
        
    if(iterations < 2):
        raise TypeError('iterations must be greater than 1.')
   
    D = _dissim(data, group_pop_var, total_pop_var)
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    p_null = data.group_pop_var.sum() / data.total_pop_var.sum()
    
    
    
    Ds = np.empty(iterations)
    
    for i in np.array(range(iterations)):

        freq_sim = np.random.binomial(n = np.array([data.total_pop_var.tolist()]), 
                                      p = np.array([[p_null] * data.shape[0]]), 
                                      size = (1, data.shape[0])).tolist()[0]
        data = data.assign(group_pop_var = freq_sim)
        aux = _dissim(data, 'group_pop_var', 'total_pop_var')
        Ds[i] = aux
        
    D_star = Ds.mean()
    
    if (D >= D_star):
        Dct = (D - D_star)/(1 - D_star)
    else:
        Dct = (D - D_star)/D_star
        
    return Dct


class Modified_Dissim:
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

    Attributes
    ----------

    statistic : float
                Modified Dissimilarity Index (Dissimilarity from Carrington and Troske (1997))
        
    Examples
    --------
    In this example, we will calculate the Modified Dissimilarity Index (Dct) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(url, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','nhblk10']]
    
    The value is estimated below.
    
    >>> np.random.seed(1234)
    >>> modified_dissim_index = Modified_Dissim(df, 'nhblk10', 'pop10')
    >>> modified_dissim_index.statistic
    0.30009504639081996
     
    Notes
    -----
    Based on Carrington, William J., and Kenneth R. Troske. "On measuring segregation in samples with small units." Journal of Business & Economic Statistics 15.4 (1997): 402-409.

    """

    def __init__(self, data, group_pop_var, total_pop_var, iterations = 500):

        self.statistic = _modified_dissim(data, group_pop_var, total_pop_var, iterations)
        