"""
Bias-Corrected Dissimilarity Segregation Index
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pandas as pd
from dissimilarity import _dissim

__all__ = ['Bias_Corrected_Dissim']


def _bias_corrected_dissim(data, group_pop_var, total_pop_var, B = 500):
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

    Attributes
    ----------

    dbc : float
          Dissimilarity with Bias-Correction (bias correction from Allen, Rebecca et al. (2015))

    Notes
    -----
    Based on Allen, Rebecca, et al. "More reliable inference for the dissimilarity index of segregation." The econometrics journal 18.1 (2015): 40-66.

    """
    if(type(B) is not int):
        raise TypeError('B must be an integer')
        
    if(B < 2):
        raise TypeError('B must be greater than 1.')
   
    D = _dissim(data, group_pop_var, total_pop_var)
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    data['other_group_pop'] = data.total_pop_var - data.group_pop_var
    
    # Group 0: minority group
    p0_i = (data.group_pop_var / data.group_pop_var.sum())
    n0 = data.group_pop_var.sum()
    sim0 = np.random.multinomial(n0, p0_i, size = B)
    
    # Group 1: complement group
    p1_i = (data.other_group_pop / data.other_group_pop.sum())
    n1 = data.other_group_pop.sum()
    sim1 = np.random.multinomial(n1, p1_i, size = B)
    
    
    Dbcs = np.empty(B)
    for i in np.array(range(B)):
        data_aux = {'simul_group': sim0[i].tolist(), 'simul_tot': (sim0[i] + sim1[i]).tolist()}
        df_aux = pd.DataFrame.from_dict(data_aux)
        Dbcs[i] = _dissim(df_aux, 'simul_group', 'simul_tot')
        
    Db = Dbcs.mean()
    
    Dbc = 2 * D - Db
    Dbc # It expected to be lower than D, because D is upwarded biased
        
    return Dbc


class Bias_Corrected_Dissim:
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

    Attributes
    ----------

    dbc : float
          Dissimilarity with Bias-Correction (bias correction from Allen, Rebecca et al. (2015))
        
    Examples
    --------
    In this example, we will calculate the Dissimilarity with Bias Correction (Dbc) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(url, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['pop10','nhblk10']]
    
    The value is estimated below.
    
    >>> np.random.seed(1234)
    >>> bias_corrected_dissim_index = Bias_Corrected_Dissim(df, 'nhblk10', 'pop10')
    >>> bias_corrected_dissim_index.dbc
    0.31484636081876954
     
    Notes
    -----
    Based on Allen, Rebecca, et al. "More reliable inference for the dissimilarity index of segregation." The econometrics journal 18.1 (2015): 40-66.

    """

    def __init__(self, data, group_pop_var, total_pop_var, B = 500):

        self.dbc = _bias_corrected_dissim(data, group_pop_var, total_pop_var, B)
        
        
    @property
    def _statistic(self):
        """More consistent hidden attribute to access Segregation statistics"""
        return self.dbc