"""
Concentration Profile based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pandas as pd
import warnings

__all__ = ['Con_Prof']


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

    statistic : float
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
    
    return R, grid, curve


class Con_Prof:
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

    statistic : float
                Concentration Profile Index
        
    Examples
    --------
    In this example, we will calculate the concentration profile (R) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
	>>> This example uses all census data that the user must provide your own copy of the external database.
	>>> A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/osnap/tree/master/osnap/data.
	>>>	After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','nhblk10']]
    
    The value is estimated below.
    
    >>> conprof_index = Con_Prof(df, 'nhblk10', 'pop10')
    >>> conprof_index.statistic
    0.06393365660089256
    
    You can plot the profile curve with the plot method.
    
    >>> conprof_index.plot()
        
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """

    def __init__(self, data, group_pop_var, total_pop_var, m = 1000):

        self.statistic = _conprof(data, group_pop_var, total_pop_var, m)[0]
        self.grid      = _conprof(data, group_pop_var, total_pop_var, m)[1]
        self.curve     = _conprof(data, group_pop_var, total_pop_var, m)[2]

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
