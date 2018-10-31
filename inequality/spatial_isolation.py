"""
Spatial Isolation based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

__all__ = ['Spatial_Isolation']


def _spatial_isolation(data, group_pop_var, total_pop_var, alpha = 0.6, beta = 0.5):
    """
    Calculation of Spatial Isolation index

    Parameters
    ----------

    data          : a geopandas DataFrame with a 'geometry' column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5

    Attributes
    ----------

    statistic : float
                Spatial Isolation Index

    Notes
    -----
    This measure is also called the distance decay isolation. It may be interpreted as the probability that the next person a group member meets anywhere in space is from the same group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.

    """
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
        
    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')
    
    if (beta < 0):
        raise ValueError('beta must be greater than zero.')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    if any(data.total_pop_var < data.group_pop_var):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
    
    # If a unit has zero population, the group of interest frequency is zero
    data = data.assign(xi = data.group_pop_var,
                       ti = data.total_pop_var,
                       c_lons = data.centroid.map(lambda p: p.x),
                       c_lats = data.centroid.map(lambda p: p.y))
    
    X = data.xi.sum()
    
    dist = euclidean_distances(data[['c_lons','c_lats']])
    np.fill_diagonal(dist, val = (alpha * data.area) ** (beta))
    c = np.exp(-dist)
    
    Pij  = np.multiply(c, np.array(data['ti'])) / np.sum(np.multiply(c, np.array(data['ti'])), axis = 1)
    SxPx = (np.array(data['xi'] / X) * np.sum(np.multiply(Pij, np.array(data['xi'] / data['ti'])), axis = 1)).sum()
    
    return SxPx


class Spatial_Isolation:
    """
    Calculation of Spatial Isolation index

    Parameters
    ----------

    data          : a geopandas DataFrame with a 'geometry' column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5

    Attributes
    ----------

    statistic : float
                Spatial Isolation Index
        
    Examples
    --------
    In this example, we will calculate the spatial isolation index (SxPx) for the Riverside County using the census tract data of 2010.
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
    
    The value is estimated below.
    
    >>> spatial_isolation_index = Spatial_Isolation(gdf, 'nhblk10', 'pop10')
    >>> spatial_isolation_index.statistic
    0.07214112078134231
            
    Notes
    -----
    This measure is also called the distance decay isolation. It may be interpreted as the probability that the next person a group member meets anywhere in space is from the same group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, alpha = 0.6, beta = 0.5):

        self.statistic = _spatial_isolation(data, group_pop_var, total_pop_var, alpha, beta)