"""
Density-Corrected Dissimilarity Segregation Index
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

__all__ = ['Density_Corrected_Dissim']


def _density_corrected_dissim(data, group_pop_var, total_pop_var):
    """
    Calculation of Density Corrected Dissimilarity index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                                  
    Attributes
    ----------

    statistic : float
                Dissimilarity with Density-Correction (density correction from Allen, Rebecca et al. (2015))

    Notes
    -----
    Based on Allen, Rebecca, et al. "More reliable inference for the dissimilarity index of segregation." The econometrics journal 18.1 (2015): 40-66.

    """
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
        
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    if any(data.total_pop_var < data.group_pop_var):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
    
    data['other_group_pop'] = data.total_pop_var - data.group_pop_var
    
    # Group 0: minority group
    p0_i = (data.group_pop_var / data.group_pop_var.sum())
    n0 = data.group_pop_var.sum()
    
    # Group 1: complement group
    p1_i = (data.other_group_pop / data.other_group_pop.sum())
    n1 = data.other_group_pop.sum()
    
    sigma_hat_j = np.sqrt(((p1_i * (1 - p1_i)) / n1) + ((p0_i * (1 - p0_i)) / n0))
    theta_hat_j = abs(p1_i - p0_i) / sigma_hat_j
    
    # Constructing function that returns $n(\hat{\theta}_j)$
    def return_optimal_theta(theta_j):
        
        def fold_norm(x):
            y = (-1) * norm.pdf(x - theta_j) + norm.pdf(x + theta_j)
            return y
        
        initial_guesses = np.array(0)
        res = minimize(fold_norm, 
                       initial_guesses, 
                       method='nelder-mead',
                       options = {'xtol': 1e-8})
        return res.final_simplex[0][1][0]
        
    optimal_thetas = theta_hat_j.apply(return_optimal_theta)
    Ddc = (sigma_hat_j * optimal_thetas).sum() / 2
    
    return Ddc


class Density_Corrected_Dissim:
    """
    Calculation of Density Corrected Dissimilarity index

    Parameters
    ----------

    data          : a pandas DataFrame
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                                  
    Attributes
    ----------

    statistic : float
                Dissimilarity with Density-Correction (density correction from Allen, Rebecca et al. (2015))
        
    Examples
    --------
    In this example, we will calculate the Dissimilarity with Density Correction (Ddc) for the Riverside County using the census tract data of 2010.
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
    
    >>> density_corrected_dissim_index = Density_Corrected_Dissim(df, 'nhblk10', 'pop10')
    >>> density_corrected_dissim_index.statistic
    0.2954881746588274
     
    Notes
    -----
    Based on Allen, Rebecca, et al. "More reliable inference for the dissimilarity index of segregation." The econometrics journal 18.1 (2015): 40-66.

    """

    def __init__(self, data, group_pop_var, total_pop_var):

        self.statistic = _density_corrected_dissim(data, group_pop_var, total_pop_var)
    