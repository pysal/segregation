"""
Relative Concentration based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import numpy as np
import pandas as pd

__all__ = ['Relative_Concentration']


def _relative_concentration(data, group_pop_var, total_pop_var):
    """
    Calculation of Relative Concentration index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Relative Concentration Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
        
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)
    
    area = np.array(data.area)
    
    if any(t < x):    
        raise ValueError('Group of interest population must equal or lower than the total population of the units.')
    
    y = t - x
    
    X = x.sum()
    Y = y.sum()
    T = t.sum()
    
    # Create the indexes according to the area ordering
    des_ind = (-area).argsort()
    asc_ind = area.argsort()
    
    n1 = np.where(((np.cumsum(t[asc_ind]) / T) < X/T) == False)[0][0]
    n2 = np.where(((np.cumsum(t[des_ind]) / T) < X/T) == False)[0][0]
    
    n  = data.shape[0]
    T1 = t[asc_ind][0:(n1+1)].sum()
    T2 = t[asc_ind][n2:n].sum()
    
    RCO = ((((x[asc_ind] * area[asc_ind] / X).sum()) / ((y[asc_ind] * area[asc_ind] / Y).sum())) - 1) / \
          ((((t[asc_ind] * area[asc_ind])[0:(n1+1)].sum() / T1) / ((t[asc_ind] * area[asc_ind])[n2:n].sum() / T2)) - 1)
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return RCO, core_data


class Relative_Concentration:
    """
    Calculation of Relative Concentration index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Attributes
    ----------

    statistic : float
                Relative Concentration Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
       
    Examples
    --------
    In this example, we will calculate the relative concentration index (RCO) for the Riverside County using the census tract data of 2010.
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
    
    >>> relative_concentration_index = Relative_Concentration(gdf, 'nhblk10', 'pop10')
    >>> relative_concentration_index.statistic
    0.5364305924831142
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    """

    def __init__(self, data, group_pop_var, total_pop_var):
        
        aux = _relative_concentration(data, group_pop_var, total_pop_var)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _relative_concentration