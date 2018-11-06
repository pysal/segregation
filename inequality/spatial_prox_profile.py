"""
Spatial Proximity Profile based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pandas as pd
import warnings
from libpysal.weights import Queen
from numpy import inf
from sklearn.metrics.pairwise import manhattan_distances

__all__ = ['Spatial_Prox_Prof']


def _spatial_prox_profile(data, group_pop_var, total_pop_var, m = 1000):
    """
    Calculation of Spatial Proximity Profile

    Parameters
    ----------

    data          : a geopandas DataFrame with a 'geometry' column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    m             : int
                    a numeric value indicating the number of thresholds to be used. Default value is 1000. 
                    A large value of m creates a smoother-looking graph and a more precise spatial proximity profile value but slows down the calculation speed.

    Attributes
    ----------

    statistic : float
                Spatial Proximity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.

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
        
    wij = Queen.from_dataframe(data).full()[0]
    delta = manhattan_distances(wij)
    
    def calculate_etat(t):
        g_t_i = np.where(data.group_pop_var / data.total_pop_var >= t, True, False)
        k = g_t_i.sum()
        sub_delta_ij = delta[g_t_i,:][:,g_t_i]
        den = sub_delta_ij.sum()
        eta_t = (k**2 - k) / den
        return eta_t
    
    grid = np.linspace(0, 1, m)
    aux = np.array(list(map(calculate_etat, grid)))
    aux[aux == inf] = 0
    aux[aux == -inf] = 0
    curve = np.nan_to_num(aux, 0)
    
    threshold = data.group_pop_var.sum() / data.total_pop_var.sum()
    SPP = ((threshold - ((curve[grid < threshold]).sum() / m - (curve[grid >= threshold]).sum()/ m)) / (1 - threshold))
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return SPP, grid, curve, core_data


class Spatial_Prox_Prof:
    """
    Calculation of Spatial Proximity Profile

    Parameters
    ----------

    data          : a geopandas DataFrame with a 'geometry' column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    m             : int
                    a numeric value indicating the number of thresholds to be used. Default value is 1000. 
                    A large value of m creates a smoother-looking graph and a more precise spatial proximity profile value but slows down the calculation speed.

    Attributes
    ----------

    statistic : float
                Spatial Proximity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the spatial proximity profile (SPP) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> This example uses all census data that the user must provide your own copy of the external database.
    >>> A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/osnap/tree/master/osnap/data.
    >>> After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['trtid10', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'trtid10')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    >>> spat_prox_index = Spatial_Prox_Prof(gdf, 'nhblk10', 'pop10')
    >>> spat_prox_index.statistic
    0.09112912525931849
    
    You can plot the profile curve with the plot method.
    
    >>> spat_prox_index.plot()
        
    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.

    """

    def __init__(self, data, group_pop_var, total_pop_var, m = 1000):
        
        aux = _spatial_prox_profile(data, group_pop_var, total_pop_var, m)

        self.statistic = aux[0]
        self.grid      = aux[1]
        self.curve     = aux[2]
        self.core_data = aux[3]
        self._function = _spatial_prox_profile

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
