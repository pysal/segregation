"""
Spatial Information Theory Index based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
import pandas as pd
import libpysal
import math
from libpysal.weights import Queen

__all__ = ['Spatial_Information_Theory']


def _spatial_information_theory(data, group_pop_var, total_pop_var, w = None, unit_in_local_env = True, original_crs = {'init': 'epsg:4326'}):
    """
    Calculation of Spatial Information Theory index

    Parameters
    ----------

    data              : a geopandas DataFrame with a geometry column.
    
    group_pop_var     : string
                        The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var     : string
                        The name of variable in data that contains the total population of the unit
                    
    w                 : W
                        A PySAL weights object. If not provided, Queen contiguity matrix is used.
                        This is used to construct the local environment around each spatial unit.
    
    unit_in_local_env : boolean
                        A condition argument that states if the local environment around the unit comprises the unit itself. Default is True.
                        
    original_crs      : the original crs code given by a dict of data, but this is later be projected for the Mercator projection (EPSG = 3395).
                        This argument is also to avoid passing data without crs and, therefore, raising unusual results.
                        This index rely on the population density and we consider the area using squared kilometers. 

    Attributes
    ----------

    statistic : float
                Spatial Information Theory Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Based on Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    
    This measure can be extended to a society with more than two groups.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError('data is not a GeoDataFrame and, therefore, this index cannot be calculated.')
    
    if((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')
    
    if ((group_pop_var not in data.columns) or (total_pop_var not in data.columns)):    
        raise ValueError('group_pop_var and total_pop_var must be variables of data')
    
    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis = 1)
        data = data.set_geometry('geometry')
        
    if w is None:    
        w_object = Queen.from_dataframe(data)
    else:
        w_object = w
    
    if (not issubclass(type(w_object), libpysal.weights.W)):
        raise TypeError('w is not a PySAL weights object')
    
    data = data.rename(columns={group_pop_var: 'group_pop_var', 
                                total_pop_var: 'total_pop_var'})
    
    data['compl_pop_var'] = data['total_pop_var'] - data['group_pop_var']
    
    
    # In this case, M = 2 according to Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    pi_1 = data['group_pop_var'].sum() / data['total_pop_var'].sum()
    pi_2 = data['compl_pop_var'].sum() / data['total_pop_var'].sum()
    E = -1 * (pi_1 * math.log(pi_1, 2) + pi_2 * math.log(pi_2, 2))
    T = data['total_pop_var'].sum()
    
    # Here you reproject the data using the Mercator projection
    data.crs = original_crs
    data = data.to_crs(crs = {'init': 'epsg:3395'})  # Mercator
    sqm_to_sqkm = 10 ** 6
    data['area_sq_km'] = data.area / sqm_to_sqkm
    tau_p = data['total_pop_var'] / data['area_sq_km']
    
    w_matrix = w_object.full()[0]

    if unit_in_local_env:
        np.fill_diagonal(w_matrix, 1)

    # The local context of each spatial unit is given by the aggregate context (this multiplication gives the local sum of each population)
    data['local_group_pop_var'] = np.matmul(data['group_pop_var'], w_matrix)
    data['local_compl_pop_var'] = np.matmul(data['compl_pop_var'], w_matrix)
    data['local_total_pop_var'] = np.matmul(data['total_pop_var'], w_matrix)
    
    pi_tilde_p_1 = np.array(data['local_group_pop_var'] / data['local_total_pop_var'])
    pi_tilde_p_2 = np.array(data['local_compl_pop_var'] / data['local_total_pop_var'])
    
    E_tilde_p = -1 * (pi_tilde_p_1 * np.log(pi_tilde_p_1) / np.log(2) + pi_tilde_p_2 * np.log(pi_tilde_p_2) / np.log(2))
    
    SIT = 1 - 1 / (T * E) * (tau_p * E_tilde_p).sum() # This is the H_Tilde according to Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    
    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]
    
    return SIT, core_data


class Spatial_Information_Theory:
    """
    Calculation of Spatial Information Theory index

    Parameters
    ----------

    data              : a geopandas DataFrame with a geometry column.
    
    group_pop_var     : string
                        The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var     : string
                        The name of variable in data that contains the total population of the unit
                    
    w                 : W
                        A PySAL weights object. If not provided, Queen contiguity matrix is used.
                        This is used to construct the local environment around each spatial unit.
    
    unit_in_local_env : boolean
                        A condition argument that states if the local environment around the unit comprises the unit itself. Default is True.

    original_crs      : the original crs code given by a dict of data, but this is later be projected for the Mercator projection (EPSG = 3395).
                        This argument is also to avoid passing data without crs and, therefore, raising unusual results.
                        This index rely on the population density and we consider the area using squared kilometers.

    Attributes
    ----------

    statistic : float
                Spatial Information Theory Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Examples
    --------
    In this example, we will calculate the degree of spatial information theory (SIT) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset. The neighborhood contiguity matrix is used.
    
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
    
    >>> spatial_information_theory_index = Spatial_Information_Theory(gdf, 'nhblk10', 'pop10')
    >>> spatial_information_theory_index.statistic
    0.789200592777201
        
    To use different neighborhood matrices:
        
    >>> from libpysal.weights import Rook, KNN
    
    Assuming K-nearest neighbors with k = 4
    
    >>> knn = KNN.from_dataframe(gdf, k=4)
    >>> spatial_information_theory_index = Spatial_Information_Theory(gdf, 'nhblk10', 'pop10', w = knn)
    >>> spatial_information_theory_index.statistic
    0.7879736633559175
    
    Assuming Rook contiguity neighborhood
    
    >>> roo = Rook.from_dataframe(gdf)
    >>> spatial_information_theory_index = Spatial_Information_Theory(gdf, 'nhblk10', 'pop10', w = roo)
    >>> spatial_information_theory_index.statistic
    0.7891079229776317
            
    Notes
    -----
    Based on Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, w = None, unit_in_local_env = True):
        
        aux = _spatial_information_theory(data, group_pop_var, total_pop_var, w, unit_in_local_env)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _spatial_information_theory