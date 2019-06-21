"""
Spatial based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
import libpysal

from libpysal.weights import Queen, KNN, insert_diagonal, lag_spatial
from numpy import inf
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from scipy.ndimage.interpolation import shift

from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix

from segregation.aspatial.aspatial_indexes import _dissim
from segregation.aspatial.multigroup_aspatial_indexes import Multi_Information_Theory
from segregation.network import calc_access
from libpysal.weights.util import attach_islands


def _localize(data, w):
    new_data = []
    w = insert_diagonal(w)
    for y in data:
        new_data.append(lag_spatial(w, y))
    return new_data


def _return_length_weighted_w(data):
    """
    Returns a PySAL weights object that the weights represent the length of the commom boudary of two areal units that share border.
    Author: Levi Wolf <levi.john.wolf@gmail.com>. 
    Thank you, Levi!

    Parameters
    ----------

    data          : a geopandas DataFrame with a 'geometry' column.

    Notes
    -----
    Currently it's not making any projection.

    """

    w = libpysal.weights.Rook.from_dataframe(
        data, ids=data.index.tolist(), geom_col=data._geometry_column_name)

    if (len(w.islands) == 0):
        w = w
    else:
        warnings('There are some islands in the GeoDataFrame.')
        w_aux = libpysal.weights.KNN.from_dataframe(
            data,
            ids=data.index.tolist(),
            geom_col=data._geometry_column_name,
            k=1)
        w = attach_islands(w, w_aux)

    adjlist = w.to_adjlist()
    islands = pd.DataFrame.from_records([{
        'focal': island,
        'neighbor': island,
        'weight': 0
    } for island in w.islands])
    merged = adjlist.merge(data.geometry.to_frame('geometry'), left_on='focal',
                           right_index=True, how='left')\
                    .merge(data.geometry.to_frame('geometry'), left_on='neighbor',
                           right_index=True, how='left', suffixes=("_focal", "_neighbor"))\

    # Transforming from pandas to geopandas
    merged = gpd.GeoDataFrame(merged, geometry='geometry_focal')
    merged['geometry_neighbor'] = gpd.GeoSeries(merged.geometry_neighbor)

    # Getting the shared boundaries
    merged['shared_boundary'] = merged.geometry_focal.intersection(
        merged.set_geometry('geometry_neighbor'))

    # Putting it back to a matrix
    merged['weight'] = merged.set_geometry('shared_boundary').length
    merged_with_islands = pd.concat((merged, islands))
    length_weighted_w = libpysal.weights.W.from_adjlist(
        merged_with_islands[['focal', 'neighbor', 'weight']])
    for island in w.islands:
        length_weighted_w.neighbors[island] = []
        del length_weighted_w.weights[island]

    length_weighted_w._reset()

    return length_weighted_w


__all__ = [
    'Spatial_Prox_Prof', 'Spatial_Dissim', 'Boundary_Spatial_Dissim',
    'Perimeter_Area_Ratio_Spatial_Dissim', 'Distance_Decay_Isolation',
    'Distance_Decay_Exposure', 'Spatial_Proximity', 'Absolute_Clustering',
    'Relative_Clustering', 'Delta', 'Absolute_Concentration',
    'Relative_Concentration', 'Absolute_Centralization',
    'Relative_Centralization', 'SpatialInformationTheory'
]


def _spatial_prox_profile(data, group_pop_var, total_pop_var, m=1000):
    """
    Calculation of Spatial Proximity Profile

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    m             : int
                    a numeric value indicating the number of thresholds to be used. Default value is 1000. 
                    A large value of m creates a smoother-looking graph and a more precise spatial proximity profile value but slows down the calculation speed.

    Returns
    ----------

    statistic : float
                Spatial Proximity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.
    
    Reference: :cite:`hong2014measuring`.

    """

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if (type(m) is not int):
        raise TypeError('m must be a string.')

    if (m < 2):
        raise ValueError('m must be greater than 1.')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    if any(data.total_pop_var < data.group_pop_var):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    # Create the shortest distance path between two pair of units using Shimbel matrix. This step was well discussed in https://github.com/pysal/segregation/issues/5.
    w_libpysal = Queen.from_dataframe(data)
    graph = csr_matrix(w_libpysal.full()[0])
    delta = floyd_warshall(csgraph=graph, directed=False)

    def calculate_etat(t):
        g_t_i = np.where(data.group_pop_var / data.total_pop_var >= t, True,
                         False)
        k = g_t_i.sum()
        sub_delta_ij = delta[
            g_t_i, :][:,
                      g_t_i]  # i and j only varies in the units subset within the threshold in eta_t of Hong (2014).
        den = sub_delta_ij.sum()
        eta_t = (k**2 - k) / den
        return eta_t

    grid = np.linspace(0, 1, m)
    aux = np.array(list(map(calculate_etat, grid)))
    aux[aux == inf] = 0
    aux[aux == -inf] = 0
    curve = np.nan_to_num(aux, 0)

    threshold = data.group_pop_var.sum() / data.total_pop_var.sum()
    SPP = ((threshold - ((curve[grid < threshold]).sum() / m -
                         (curve[grid >= threshold]).sum() / m)) /
           (1 - threshold))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return SPP, grid, curve, core_data


class Spatial_Prox_Prof:
    """
    Calculation of Spatial Proximity Profile

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
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
                Spatial Proximity Profile Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the spatial proximity profile (SPP) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Spatial_Prox_Prof
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    >>> spat_prox_index = Spatial_Prox_Prof(gdf, 'nhblk10', 'pop10')
    >>> spat_prox_index.statistic
    0.11217269612149207
    
    You can plot the profile curve with the plot method.
    
    >>> spat_prox_index.plot()
        
    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.
    
    Reference: :cite:`hong2014measuring`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, m=1000):

        aux = _spatial_prox_profile(data, group_pop_var, total_pop_var, m)

        self.statistic = aux[0]
        self.grid = aux[1]
        self.curve = aux[2]
        self.core_data = aux[3]
        self._function = _spatial_prox_profile

    def plot(self):
        """
        Plot the Spatial Proximity Profile
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn('This method relies on importing `matplotlib`')
        graph = plt.scatter(self.grid, self.curve, s=0.1)
        return graph


def _spatial_dissim(data,
                    group_pop_var,
                    total_pop_var,
                    w=None,
                    standardize=False):
    """
    Calculation of Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    w             : W
                    A PySAL weights object. If not provided, Queen contiguity matrix is used.
                    
    standardize   : boolean
                    A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
                    For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
                    works by default with row standardization.
        
    Returns
    ----------

    statistic : float
                Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Based on Morrill, R. L. (1991) "On the Measure of Geographic Segregation". Geography Research Forum.
    
    Reference: :cite:`morrill1991measure`.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if (type(standardize) is not bool):
        raise TypeError('std is not a boolean object')

    if w is None:
        w_object = Queen.from_dataframe(data)
    else:
        w_object = w

    if (not issubclass(type(w_object), libpysal.weights.W)):
        raise TypeError('w is not a PySAL weights object')

    D = _dissim(data, group_pop_var, total_pop_var)[0]

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    # If a unit has zero population, the group of interest frequency is zero
    pi = np.where(t == 0, 0, x / t)

    if not standardize:
        cij = w_object.full()[0]
    else:
        cij = w_object.full()[0]
        cij = cij / cij.sum(axis=1).reshape((cij.shape[0], 1))

    # Inspired in (second solution): https://stackoverflow.com/questions/22720864/efficiently-calculating-a-euclidean-distance-matrix-using-numpy
    # Distance Matrix
    abs_dist = abs(pi[..., np.newaxis] - pi)

    # manhattan_distances used to compute absolute distances
    num = np.multiply(abs_dist, cij).sum()
    den = cij.sum()
    SD = D - num / den
    SD

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return SD, core_data


class Spatial_Dissim:
    """
    Calculation of Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    w             : W
                    A PySAL weights object. If not provided, Queen contiguity matrix is used.
    
    standardize   : boolean
                    A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
                    For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
                    works by default with row standardization.

    Attributes
    ----------

    statistic : float
                Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.   
                
    Examples
    --------
    In this example, we will calculate the degree of spatial dissimilarity (D) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset. The neighborhood contiguity matrix is used.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Spatial_Dissim
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_dissim_index = Spatial_Dissim(gdf, 'nhblk10', 'pop10')
    >>> spatial_dissim_index.statistic
    0.2864885055405311
        
    To use different neighborhood matrices:
        
    >>> from libpysal.weights import Rook, KNN
    
    Assuming K-nearest neighbors with k = 4
    
    >>> knn = KNN.from_dataframe(gdf, k=4)
    >>> spatial_dissim_index = Spatial_Dissim(gdf, 'nhblk10', 'pop10', w = knn)
    >>> spatial_dissim_index.statistic
    0.28544347200877285
    
    Assuming Rook contiguity neighborhood
    
    >>> roo = Rook.from_dataframe(gdf)
    >>> spatial_dissim_index = Spatial_Dissim(gdf, 'nhblk10', 'pop10', w = roo)
    >>> spatial_dissim_index.statistic
    0.2866269198707091
            
    Notes
    -----
    Based on Morrill, R. L. (1991) "On the Measure of Geographic Segregation". Geography Research Forum.
    
    Reference: :cite:`morrill1991measure`.
    
    """

    def __init__(self,
                 data,
                 group_pop_var,
                 total_pop_var,
                 w=None,
                 standardize=False):

        aux = _spatial_dissim(data, group_pop_var, total_pop_var, w,
                              standardize)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _spatial_dissim


def _boundary_spatial_dissim(data,
                             group_pop_var,
                             total_pop_var,
                             standardize=False):
    """
    Calculation of Boundary Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    standardize   : boolean
                    A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
                    For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
                    works by default without row standardization. That is, directly with border length.
        
    Returns
    ----------

    statistic : float
                Boundary Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    The formula is based on Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
    
    Original paper by Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.
    
    References: :cite:`hong2014implementing` and :cite:`wong1993spatial`.

    """

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if (type(standardize) is not bool):
        raise TypeError('std is not a boolean object')

    D = _dissim(data, group_pop_var, total_pop_var)[0]

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    # If a unit has zero population, the group of interest frequency is zero
    data = data.assign(
        pi=np.where(data.total_pop_var == 0, 0, data.group_pop_var /
                    data.total_pop_var))

    if not standardize:
        cij = _return_length_weighted_w(data).full()[0]
    else:
        cij = _return_length_weighted_w(data).full()[0]
        cij = cij / cij.sum(axis=1).reshape((cij.shape[0], 1))

    # manhattan_distances used to compute absolute distances
    num = np.multiply(manhattan_distances(data[['pi']]), cij).sum()
    den = cij.sum()
    BSD = D - num / den
    BSD

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return BSD, core_data


class Boundary_Spatial_Dissim:
    """
    Calculation of Boundary Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    standardize   : boolean
                    A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
                    For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
                    works by default without row standardization. That is, directly with border length.
        

    Attributes
    ----------

    statistic : float
                Boundary Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
         
    Examples
    --------
    In this example, we will calculate the degree of boundary spatial dissimilarity (D) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Boundary_Spatial_Dissim
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> boundary_spatial_dissim_index = Boundary_Spatial_Dissim(gdf, 'nhblk10', 'pop10')
    >>> boundary_spatial_dissim_index.statistic
    0.28869903953453163
            
    Notes
    -----
    The formula is based on Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
    
    Original paper by Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.
    
    References: :cite:`hong2014implementing` and :cite:`wong1993spatial`.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, standardize=False):

        aux = _boundary_spatial_dissim(data, group_pop_var, total_pop_var,
                                       standardize)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _boundary_spatial_dissim


def _perimeter_area_ratio_spatial_dissim(data,
                                         group_pop_var,
                                         total_pop_var,
                                         standardize=True):
    """
    Calculation of Perimeter/Area Ratio Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    standardize   : boolean
                    A condition for standardisation of the weights matrices. 
                    If True, the values of cij in the formulas gets standardized and the overall sum is 1.

    Returns
    ----------

    statistic : float
                Perimeter/Area Ratio Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Originally based on Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.
    
    However, Tivadar, Mihai. "OasisR: An R Package to Bring Some Order to the World of Segregation Measurement." Journal of Statistical Software 89.1 (2019): 1-39.
    points out that in Wong’s original there is an issue with the formula which is an extra division by 2 in the spatial interaction component.
    This function follows the formula present in the first Appendix of Tivadar, Mihai. "OasisR: An R Package to Bring Some Order to the World of Segregation Measurement." Journal of Statistical Software 89.1 (2019): 1-39.

    References: :cite:`wong1993spatial` and :cite:`tivadar2019oasisr`.
        
    """

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if (type(standardize) is not bool):
        raise TypeError('std is not a boolean object')

    D = _dissim(data, group_pop_var, total_pop_var)[0]

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    # If a unit has zero population, the group of interest frequency is zero
    data = data.assign(
        pi=np.where(data.total_pop_var == 0, 0, data.group_pop_var /
                    data.total_pop_var))

    if not standardize:
        cij = _return_length_weighted_w(data).full()[0]
    else:
        cij = _return_length_weighted_w(data).full()[0]
        cij = cij / cij.sum()

    peri = data.length
    ai = data.area

    aux_sum = np.add(
        np.array(list((peri / ai))),
        np.array(list((peri / ai))).reshape((len(list((peri / ai))), 1)))

    max_pa = max(peri / ai)

    num = np.multiply(np.multiply(manhattan_distances(data[['pi']]), cij),
                      aux_sum).sum()
    den = 2 * max_pa

    PARD = D - (num / den)
    PARD

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return PARD, core_data


class Perimeter_Area_Ratio_Spatial_Dissim:
    """
    Calculation of Perimeter/Area Ratio Spatial Dissimilarity index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    standardize   : boolean
                    A condition for standardisation of the weights matrices. 
                    If True, the values of cij in the formulas gets standardized and the overall sum is 1.
        
    Attributes
    ----------

    statistic : float
                Perimeter/Area Ratio Spatial Dissimilarity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.      
                
    Examples
    --------
    In this example, we will calculate the degree of perimeter/area ratio spatial dissimilarity (PARD) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Perimeter_Area_Ratio_Spatial_Dissim
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> perimeter_area_ratio_spatial_dissim_index = Perimeter_Area_Ratio_Spatial_Dissim(gdf, 'nhblk10', 'pop10')
    >>> perimeter_area_ratio_spatial_dissim_index.statistic
    0.31260876347432687
            
    Notes
    -----
    Originally based on Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.
    
    However, Tivadar, Mihai. "OasisR: An R Package to Bring Some Order to the World of Segregation Measurement." Journal of Statistical Software 89.1 (2019): 1-39.
    points out that in Wong’s original there is an issue with the formula which is an extra division by 2 in the spatial interaction component.
    This function follows the formula present in the first Appendix of Tivadar, Mihai. "OasisR: An R Package to Bring Some Order to the World of Segregation Measurement." Journal of Statistical Software 89.1 (2019): 1-39.
    
    References: :cite:`wong1993spatial` and :cite:`tivadar2019oasisr`.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, standardize=True):

        aux = _perimeter_area_ratio_spatial_dissim(data, group_pop_var,
                                                   total_pop_var, standardize)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _perimeter_area_ratio_spatial_dissim


def _distance_decay_isolation(data,
                              group_pop_var,
                              total_pop_var,
                              alpha=0.6,
                              beta=0.5):
    """
    Calculation of Distance Decay Isolation index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5

    Returns
    ----------

    statistic : float
                Distance Decay Isolation Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    It may be interpreted as the probability that the next person a group member meets anywhere in space is from the same group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`morgan1983distance`.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')

    if (beta < 0):
        raise ValueError('beta must be greater than zero.')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    X = x.sum()

    dist = euclidean_distances(
        pd.DataFrame({
            'c_lons': c_lons,
            'c_lats': c_lats
        }))
    np.fill_diagonal(dist, val=(alpha * data.area)**(beta))
    c = np.exp(-dist)

    Pij = np.multiply(c, t) / np.sum(np.multiply(c, t), axis=1)
    DDxPx = (np.array(x / X) *
             np.nansum(np.multiply(Pij, np.array(x / t)), axis=1)).sum()

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return DDxPx, core_data


class Distance_Decay_Isolation:
    """
    Calculation of Distance Decay Isolation index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
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
                Distance Decay Isolation Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the distance decay isolation index (DDxPx) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Distance_Decay_Isolation
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_isolation_index = Distance_Decay_Isolation(gdf, 'nhblk10', 'pop10')
    >>> spatial_isolation_index.statistic
    0.07214112078134231
            
    Notes
    -----
    It may be interpreted as the probability that the next person a group member meets anywhere in space is from the same group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`morgan1983distance`.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, alpha=0.6,
                 beta=0.5):

        aux = _distance_decay_isolation(data, group_pop_var, total_pop_var,
                                        alpha, beta)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _distance_decay_isolation


def _distance_decay_exposure(data,
                             group_pop_var,
                             total_pop_var,
                             alpha=0.6,
                             beta=0.5):
    """
    Calculation of Distance Decay Exposure index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5

    Returns
    ----------

    statistic : float
                Distance Decay Exposure Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    It may be interpreted as the probability that the next person a group member meets anywhere in space is from the other group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`morgan1983distance`.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')

    if (beta < 0):
        raise ValueError('beta must be greater than zero.')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    y = t - x

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    X = x.sum()

    dist = euclidean_distances(
        pd.DataFrame({
            'c_lons': c_lons,
            'c_lats': c_lats
        }))
    np.fill_diagonal(dist, val=(alpha * data.area)**(beta))
    c = np.exp(-dist)

    Pij = np.multiply(c, t) / np.sum(np.multiply(c, t), axis=1)
    DDxPy = (x / X * np.nansum(np.multiply(Pij, y / t), axis=1)).sum()

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return DDxPy, core_data


class Distance_Decay_Exposure:
    """
    Calculation of Distance Decay Exposure index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
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
                Distance Decay Exposure Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the distance decay exposure index (DDxPy) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Distance_Decay_Exposure
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_exposure_index = Distance_Decay_Exposure(gdf, 'nhblk10', 'pop10')
    >>> spatial_exposure_index.statistic
    0.9605053172501217
            
    Notes
    -----
    It may be interpreted as the probability that the next person a group member meets anywhere in space is from the other group.
    
    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`morgan1983distance`.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, alpha=0.6,
                 beta=0.5):

        aux = _distance_decay_exposure(data, group_pop_var, total_pop_var,
                                       alpha, beta)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _distance_decay_exposure


def _spatial_proximity(data, group_pop_var, total_pop_var, alpha=0.6,
                       beta=0.5):
    """
    Calculation of Spatial Proximity index
    
    Parameters
    ----------
    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5
                    
    Returns
    ----------
    statistic : float
                Spatial Proximity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')

    if (beta < 0):
        raise ValueError('beta must be greater than zero.')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    if any(data.total_pop_var < data.group_pop_var):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    T = data.total_pop_var.sum()

    data = data.assign(xi=data.group_pop_var,
                       yi=data.total_pop_var - data.group_pop_var,
                       ti=data.total_pop_var,
                       c_lons=data.centroid.map(lambda p: p.x),
                       c_lats=data.centroid.map(lambda p: p.y))

    X = data.xi.sum()
    Y = data.yi.sum()

    dist = euclidean_distances(data[['c_lons', 'c_lats']])
    np.fill_diagonal(dist, val=(alpha * data.area)**(beta))
    c = np.exp(-dist)

    Pxx = ((np.array(data.xi) * c).T * np.array(data.xi)).sum() / X**2
    Pyy = ((np.array(data.yi) * c).T * np.array(data.yi)).sum() / Y**2
    Ptt = ((np.array(data.ti) * c).T * np.array(data.ti)).sum() / T**2
    SP = (X * Pxx + Y * Pyy) / (T * Ptt)

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return SP, core_data


class Spatial_Proximity:
    """
    Calculation of Spatial Proximity index
    
    Parameters
    ----------
    data          : a geopandas DataFrame with a geometry column.
    
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
                Spatial Proximity Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the degree of spatial proximity (SP) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Spatial_Proximity
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> spatial_prox_index = Spatial_Proximity(gdf, 'nhblk10', 'pop10')
    >>> spatial_prox_index.statistic
    1.002191883006537
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, alpha=0.6,
                 beta=0.5):

        aux = _spatial_proximity(data, group_pop_var, total_pop_var, alpha,
                                 beta)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _spatial_proximity


def _absolute_clustering(data,
                         group_pop_var,
                         total_pop_var,
                         alpha=0.6,
                         beta=0.5):
    """
    Calculation of Absolute Clustering index
    
    Parameters
    ----------
    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5
                    
    Returns
    ----------
    statistic : float
                Absolute Clustering Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')

    if (beta < 0):
        raise ValueError('beta must be greater than zero.')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    if any(data.total_pop_var < data.group_pop_var):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    data = data.assign(xi=data.group_pop_var,
                       yi=data.total_pop_var - data.group_pop_var,
                       c_lons=data.centroid.map(lambda p: p.x),
                       c_lats=data.centroid.map(lambda p: p.y))

    X = data.xi.sum()

    x = np.array(data.xi)
    t = np.array(data.total_pop_var)
    n = len(data)

    dist = euclidean_distances(data[['c_lons', 'c_lats']])
    np.fill_diagonal(dist, val=(alpha * data.area)**(beta))
    c = np.exp(-dist)

    ACL = ((((x/X) * (c * x).sum(axis = 1)).sum()) - ((X / n**2) * c.sum())) / \
          ((((x/X) * (c * t).sum(axis = 1)).sum()) - ((X / n**2) * c.sum()))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return ACL, core_data


class Absolute_Clustering:
    """
    Calculation of Absolute Clustering index
    
    Parameters
    ----------
    data          : a geopandas DataFrame with a geometry column.
    
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
                Absolute Clustering Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the absolute clustering measure (ACL) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
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
    
    >>> absolute_clust_index = Absolute_Clustering(gdf, 'nhblk10', 'pop10')
    >>> absolute_clust_index.statistic
    0.20979814508119624
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, alpha=0.6,
                 beta=0.5):

        aux = _absolute_clustering(data, group_pop_var, total_pop_var, alpha,
                                   beta)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _absolute_clustering


def _relative_clustering(data,
                         group_pop_var,
                         total_pop_var,
                         alpha=0.6,
                         beta=0.5):
    """
    Calculation of Relative Clustering index
    
    Parameters
    ----------
    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5
                    
    Returns
    ----------
    statistic : float
                Relative Clustering Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    if (alpha < 0):
        raise ValueError('alpha must be greater than zero.')

    if (beta < 0):
        raise ValueError('beta must be greater than zero.')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    if any(data.total_pop_var < data.group_pop_var):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    data = data.assign(xi=data.group_pop_var,
                       yi=data.total_pop_var - data.group_pop_var,
                       c_lons=data.centroid.map(lambda p: p.x),
                       c_lats=data.centroid.map(lambda p: p.y))

    X = data.xi.sum()
    Y = data.yi.sum()

    dist = euclidean_distances(data[['c_lons', 'c_lats']])
    np.fill_diagonal(dist, val=(alpha * data.area)**(beta))
    c = np.exp(-dist)

    Pxx = ((np.array(data.xi) * c).T * np.array(data.xi)).sum() / X**2
    Pyy = ((np.array(data.yi) * c).T * np.array(data.yi)).sum() / Y**2
    RCL = Pxx / Pyy - 1

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return RCL, core_data


class Relative_Clustering:
    """
    Calculation of Relative Clustering index
    
    Parameters
    ----------
    data          : a geopandas DataFrame with a geometry column.
    
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
                Relative Clustering Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the relative clustering measure (RCL) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Relative_Clustering
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> relative_clust_index = Relative_Clustering(gdf, 'nhblk10', 'pop10')
    >>> relative_clust_index.statistic
    0.12418089857347714
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.
    
    Reference: :cite:`massey1988dimensions`.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var, alpha=0.6,
                 beta=0.5):

        aux = _relative_clustering(data, group_pop_var, total_pop_var, alpha,
                                   beta)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _relative_clustering


def _delta(data, group_pop_var, total_pop_var):
    """
    Calculation of Delta index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Returns
    ----------

    statistic : float
                Delta Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    area = np.array(data.area)

    X = x.sum()
    A = area.sum()

    DEL = 1 / 2 * abs(x / X - area / A).sum()

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return DEL, core_data


class Delta:
    """
    Calculation of Delta index

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
                Delta Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
        
    Examples
    --------
    In this example, we will calculate the delta index (D) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Delta
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> delta_index = Delta(gdf, 'nhblk10', 'pop10')
    >>> delta_index.statistic
    0.8367330649317353
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.
    
    """

    def __init__(self, data, group_pop_var, total_pop_var):

        aux = _delta(data, group_pop_var, total_pop_var)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _delta


def _absolute_concentration(data, group_pop_var, total_pop_var):
    """
    Calculation of Absolute Concentration index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    Returns
    ----------

    statistic : float
                Absolute Concentration Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    area = np.array(data.area)

    X = x.sum()
    T = t.sum()

    # Create the indexes according to the area ordering
    des_ind = (-area).argsort()
    asc_ind = area.argsort()

    n1 = np.where(((np.cumsum(t[asc_ind]) / T) < X / T) == False)[0][0]
    n2 = np.where(((np.cumsum(t[des_ind]) / T) < X / T) == False)[0][0]

    n = data.shape[0]
    T1 = t[asc_ind][0:n1].sum()
    T2 = t[asc_ind][n2:n].sum()

    ACO = 1- ((((x[asc_ind] * area[asc_ind] / X).sum()) - ((t[asc_ind] * area[asc_ind] / T1)[0:n1].sum())) / \
          (((t[asc_ind] * area[asc_ind] / T2)[n2:n].sum()) - ((t[asc_ind] * area[asc_ind]/T1)[0:n1].sum())))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    return ACO, core_data


class Absolute_Concentration:
    """
    Calculation of Absolute Concentration index

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
                Absolute Concentration Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
                
    Examples
    --------
    In this example, we will calculate the absolute concentration index (ACO) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Absolute_Concentration
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> absolute_concentration_index = Absolute_Concentration(gdf, 'nhblk10', 'pop10')
    >>> absolute_concentration_index.statistic
    0.5430616390401855
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self, data, group_pop_var, total_pop_var):

        aux = _absolute_concentration(data, group_pop_var, total_pop_var)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _absolute_concentration


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

    Returns
    ----------

    statistic : float
                Relative Concentration Index
                
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    area = np.array(data.area)

    y = t - x

    X = x.sum()
    Y = y.sum()
    T = t.sum()

    # Create the indexes according to the area ordering
    des_ind = (-area).argsort()
    asc_ind = area.argsort()

    n1 = np.where(((np.cumsum(t[asc_ind]) / T) < X / T) == False)[0][0]
    n2 = np.where(((np.cumsum(t[des_ind]) / T) < X / T) == False)[0][0]

    n = data.shape[0]
    T1 = t[asc_ind][0:n1].sum()
    T2 = t[asc_ind][n2:n].sum()

    RCO = ((((x[asc_ind] * area[asc_ind] / X).sum()) / ((y[asc_ind] * area[asc_ind] / Y).sum())) - 1) / \
          ((((t[asc_ind] * area[asc_ind])[0:n1].sum() / T1) / ((t[asc_ind] * area[asc_ind])[n2:n].sum() / T2)) - 1)

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
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Relative_Concentration
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> relative_concentration_index = Relative_Concentration(gdf, 'nhblk10', 'pop10')
    >>> relative_concentration_index.statistic
    0.5364305924831142
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self, data, group_pop_var, total_pop_var):

        aux = _relative_concentration(data, group_pop_var, total_pop_var)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _relative_concentration


def _absolute_centralization(data, group_pop_var, total_pop_var,
                             center="mean"):
    """
    Calculation of Absolute Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    center        : string, two-dimension values (tuple, list, array) or integer.
                    This defines what is considered to be the center of the spatial context under study.

                    If string, this can be set to:
                        
                        "mean": the center longitude/latitude is the mean of longitudes/latitudes of all units. 
                        "median": the center longitude/latitude is the median of longitudes/latitudes of all units. 
                        "population_weighted_mean": the center longitude/latitude is the mean of longitudes/latitudes of all units weighted by the total population.
                        "largest_population": the center longitude/latitude is the centroid of the unit with largest total population. If there is a tie in the maximum population, the mean of all coordinates will be taken.
                    
                    If tuple, list or array, this argument should be the coordinates of the desired center assuming longitude as first value and latitude second value. Therefore, in the form (longitude, latitude), if tuple, or [longitude, latitude] if list or numpy array.
                    
                    If integer, the center will be the centroid of the polygon from data corresponding to the integer interpreted as index. 
                    For example, if `center = 0` the centroid of the first row of data is used as center, if `center = 1` the second row will be used, and so on.

    Returns
    ----------

    statistic     : float
                    Absolute Centralization Index
                
    core_data     : a geopandas DataFrame
                    A geopandas DataFrame that contains the columns used to perform the estimate.
    
    center_values : list
                    The center, in the form [longitude, latitude], values used for the calculation of the centralization distances.
    
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    A discussion of defining the center in this function can be found in https://github.com/pysal/segregation/issues/18.
    
    Reference: :cite:`massey1988dimensions`.

    """

    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    area = np.array(data.area)

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    if isinstance(center, str):
        if not center in [
                'mean', 'median', 'population_weighted_mean',
                'largest_population'
        ]:
            raise ValueError(
                'The center string must one of \'mean\', \'median\', \'population_weighted_mean\', \'largest_population\''
            )

        if (center == "mean"):
            center_lon = c_lons.mean()
            center_lat = c_lats.mean()

        if (center == "median"):
            center_lon = np.median(c_lons)
            center_lat = np.median(c_lats)

        if (center == "population_weighted_mean"):
            center_lon = np.average(c_lons, weights=t)
            center_lat = np.average(c_lats, weights=t)

        if (center == "largest_population"):
            center_lon = c_lons[np.where(t == t.max())].mean()
            center_lat = c_lats[np.where(t == t.max())].mean()

    if isinstance(center, tuple) or isinstance(center, list) or isinstance(
            center, np.ndarray):
        if np.array(center).shape != (2, ):
            raise ValueError('The center tuple/list/array must have length 2.')

        center_lon = center[0]
        center_lat = center[1]

    if isinstance(center, int):
        if (center > len(data) - 1) or (center < 0):
            raise ValueError('The center index must by in the range of data.')

        center_lon = data.iloc[[center]].centroid.x.values[0]
        center_lat = data.iloc[[center]].centroid.y.values[0]

    X = x.sum()
    A = area.sum()

    center_dist = np.sqrt((c_lons - center_lon)**2 + (c_lats - center_lat)**2)

    asc_ind = center_dist.argsort()

    Xi = np.cumsum(x[asc_ind]) / X
    Ai = np.cumsum(area[asc_ind]) / A

    ACE = np.nansum(shift(Xi, 1, cval=np.NaN) * Ai) - \
          np.nansum(Xi * shift(Ai, 1, cval=np.NaN))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    center_values = [center_lon, center_lat]

    return ACE, core_data, center_values


class Absolute_Centralization:
    """
    Calculation of Absolute Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
                    
    center        : string, two-dimension values (tuple, list, array) or integer.
                    This defines what is considered to be the center of the spatial context under study.

                    If string, this can be set to:
                        
                        "mean": the center longitude/latitude is the mean of longitudes/latitudes of all units. 
                        "median": the center longitude/latitude is the median of longitudes/latitudes of all units. 
                        "population_weighted_mean": the center longitude/latitude is the mean of longitudes/latitudes of all units weighted by the total population.
                        "largest_population": the center longitude/latitude is the centroid of the unit with largest total population. If there is a tie in the maximum population, the mean of all coordinates will be taken.
                    
                    If tuple, list or array, this argument should be the coordinates of the desired center assuming longitude as first value and latitude second value. Therefore, in the form (longitude, latitude), if tuple, or [longitude, latitude] if list or numpy array.
                    
                    If integer, the center will be the centroid of the polygon from data corresponding to the integer interpreted as index. 
                    For example, if `center = 0` the centroid of the first row of data is used as center, if `center = 1` the second row will be used, and so on.

    Attributes
    ----------

    statistic     : float
                    Absolute Centralization Index
                
    core_data     : a geopandas DataFrame
                    A geopandas DataFrame that contains the columns used to perform the estimate.
    
    center_values : list
                    The center, in the form [longitude, latitude], values used for the calculation of the centralization distances.
                
    Examples
    --------
    In this example, we will calculate the absolute centralization index (ACE) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Absolute_Centralization
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> absolute_centralization_index = Absolute_Centralization(gdf, 'nhblk10', 'pop10')
    >>> absolute_centralization_index.statistic
    0.6416113799795511
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    A discussion of defining the center in this function can be found in https://github.com/pysal/segregation/issues/18.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, center="mean"):

        aux = _absolute_centralization(data, group_pop_var, total_pop_var,
                                       center)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self.center_values = aux[2]
        self._function = _absolute_centralization


def _relative_centralization(data, group_pop_var, total_pop_var,
                             center="mean"):
    """
    Calculation of Relative Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    center        : string, two-dimension values (tuple, list, array) or integer.
                    This defines what is considered to be the center of the spatial context under study.

                    If string, this can be set to:
                        
                        "mean": the center longitude/latitude is the mean of longitudes/latitudes of all units. 
                        "median": the center longitude/latitude is the median of longitudes/latitudes of all units. 
                        "population_weighted_mean": the center longitude/latitude is the mean of longitudes/latitudes of all units weighted by the total population.
                        "largest_population": the center longitude/latitude is the centroid of the unit with largest total population. If there is a tie in the maximum population, the mean of all coordinates will be taken.
                    
                    If tuple, list or array, this argument should be the coordinates of the desired center assuming longitude as first value and latitude second value. Therefore, in the form (longitude, latitude), if tuple, or [longitude, latitude] if list or numpy array.
                    
                    If integer, the center will be the centroid of the polygon from data corresponding to the integer interpreted as index. 
                    For example, if `center = 0` the centroid of the first row of data is used as center, if `center = 1` the second row will be used, and so on.

    Returns
    ----------

    statistic     : float
                    Relative Centralization Index
                
    core_data     : a geopandas DataFrame
                    A geopandas DataFrame that contains the columns used to perform the estimate.
    
    center_values : list
                    The center, in the form [longitude, latitude], values used for the calculation of the centralization distances.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    A discussion of defining the center in this function can be found in https://github.com/pysal/segregation/issues/18.

    """
    if (str(type(data)) != '<class \'geopandas.geodataframe.GeoDataFrame\'>'):
        raise TypeError(
            'data is not a GeoDataFrame and, therefore, this index cannot be calculated.'
        )

    if ('geometry' not in data.columns):
        data['geometry'] = data[data._geometry_column_name]
        data = data.drop([data._geometry_column_name], axis=1)
        data = data.set_geometry('geometry')

    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data.columns)
            or (total_pop_var not in data.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data')

    data = data.rename(columns={
        group_pop_var: 'group_pop_var',
        total_pop_var: 'total_pop_var'
    })

    x = np.array(data.group_pop_var)
    t = np.array(data.total_pop_var)

    if any(t < x):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units.'
        )

    y = t - x

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    if isinstance(center, str):
        if not center in [
                'mean', 'median', 'population_weighted_mean',
                'largest_population'
        ]:
            raise ValueError(
                'The center string must one of \'mean\', \'median\', \'population_weighted_mean\', \'largest_population\''
            )

        if (center == "mean"):
            center_lon = c_lons.mean()
            center_lat = c_lats.mean()

        if (center == "median"):
            center_lon = np.median(c_lons)
            center_lat = np.median(c_lats)

        if (center == "population_weighted_mean"):
            center_lon = np.average(c_lons, weights=t)
            center_lat = np.average(c_lats, weights=t)

        if (center == "largest_population"):
            center_lon = c_lons[np.where(t == t.max())].mean()
            center_lat = c_lats[np.where(t == t.max())].mean()

    if isinstance(center, tuple) or isinstance(center, list) or isinstance(
            center, np.ndarray):
        if np.array(center).shape != (2, ):
            raise ValueError('The center tuple/list/array must have length 2.')

        center_lon = center[0]
        center_lat = center[1]

    if isinstance(center, int):
        if (center > len(data) - 1) or (center < 0):
            raise ValueError('The center index must by in the range of data.')

        center_lon = data.iloc[[center]].centroid.x.values[0]
        center_lat = data.iloc[[center]].centroid.y.values[0]

    X = x.sum()
    Y = y.sum()

    center_dist = np.sqrt((c_lons - center_lon)**2 + (c_lats - center_lat)**2)

    asc_ind = center_dist.argsort()

    Xi = np.cumsum(x[asc_ind]) / X
    Yi = np.cumsum(y[asc_ind]) / Y

    RCE = np.nansum(shift(Xi, 1, cval=np.NaN) * Yi) - \
          np.nansum(Xi * shift(Yi, 1, cval=np.NaN))

    core_data = data[['group_pop_var', 'total_pop_var', 'geometry']]

    center_values = [center_lon, center_lat]

    return RCE, core_data, center_values


class Relative_Centralization:
    """
    Calculation of Relative Centralization index

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit

    center        : string, two-dimension values (tuple, list, array) or integer.
                    This defines what is considered to be the center of the spatial context under study.

                    If string, this can be set to:
                        
                        "mean": the center longitude/latitude is the mean of longitudes/latitudes of all units. 
                        "median": the center longitude/latitude is the median of longitudes/latitudes of all units. 
                        "population_weighted_mean": the center longitude/latitude is the mean of longitudes/latitudes of all units weighted by the total population.
                        "largest_population": the center longitude/latitude is the centroid of the unit with largest total population. If there is a tie in the maximum population, the mean of all coordinates will be taken.
                    
                    If tuple, list or array, this argument should be the coordinates of the desired center assuming longitude as first value and latitude second value. Therefore, in the form (longitude, latitude), if tuple, or [longitude, latitude] if list or numpy array.
                    
                    If integer, the center will be the centroid of the polygon from data corresponding to the integer interpreted as index. 
                    For example, if `center = 0` the centroid of the first row of data is used as center, if `center = 1` the second row will be used, and so on.

    Attributes
    ----------

    statistic     : float
                    Relative Centralization Index
            
    core_data     : a geopandas DataFrame
                    A geopandas DataFrame that contains the columns used to perform the estimate.
    
    center_values : list
                    The center, in the form [longitude, latitude], values used for the calculation of the centralization distances.
        
    Examples
    --------
    In this example, we will calculate the relative centralization index (RCE) for the Riverside County using the census tract data of 2010.
    The group of interest is non-hispanic black people which is the variable nhblk10 in the dataset.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> import segregation
    >>> from segregation.spatial import Relative_Centralization
    
    Secondly, we need to read the data:
    
    >>> # This example uses all census data that the user must provide your own copy of the external database.
    >>> # A step-by-step procedure for downloading the data can be found here: https://github.com/spatialucr/geosnap/tree/master/geosnap/data.
    >>> # After the user download the LTDB_Std_All_fullcount.zip and extract the files, the filepath might be something like presented below.
    >>> filepath = '~/data/LTDB_Std_2010_fullcount.csv'
    >>> census_2010 = pd.read_csv(filepath, encoding = "ISO-8859-1", sep = ",")
    
    Then, we filter only for the desired county (in this case, Riverside County):
    
    >>> df = census_2010.loc[census_2010.county == "Riverside County"][['tractid', 'pop10','nhblk10']]
    
    Then, we read the Riverside map data using geopandas (the county id is 06065):
    
    >>> map_url = 'https://raw.githubusercontent.com/renanxcortes/inequality-segregation-supplementary-files/master/Tracts_grouped_by_County/06065.json'
    >>> map_gpd = gpd.read_file(map_url)
    
    It is necessary to harmonize the data type of the dataset and the geopandas in order to work the merging procedure.
    Later, we extract only the columns that will be used.
    
    >>> map_gpd['INTGEOID10'] = pd.to_numeric(map_gpd["GEOID10"])
    >>> gdf_pre = map_gpd.merge(df, left_on = 'INTGEOID10', right_on = 'tractid')
    >>> gdf = gdf_pre[['geometry', 'pop10', 'nhblk10']]
    
    The value is estimated below.
    
    >>> relative_centralization_index = Relative_Centralization(gdf, 'nhblk10', 'pop10')
    >>> relative_centralization_index.statistic
    0.18550429720565376
            
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.
    
    A discussion of defining the center in this function can be found in https://github.com/pysal/segregation/issues/18.
    
    Reference: :cite:`massey1988dimensions`.

    """

    def __init__(self, data, group_pop_var, total_pop_var, center="mean"):

        aux = _relative_centralization(data, group_pop_var, total_pop_var,
                                       center)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self.center_values = aux[2]
        self._function = _relative_centralization


class SpatialInformationTheory(Multi_Information_Theory):
    """Spatial Multigroup Information Theory Index.

    This class calculates the spatial version of the multigroup information
    theory index. The data are "spatialized" by converting each observation
    to a "local environment" by creating a weighted sum of the focal unit with
    its neighboring observations, where the neighborhood is defined by a libpysal
    weights matrix of a pandana Network instance.

    Parameters
    ----------
    data : geopandas.GeoDataFrame
        geodataframe with
    groups : list
        list of columns on gdf representing population groups for which the SIT
        index should be calculated
    network : pandana.Network
        pandana.Network instance. This is likely created with `get_network` or
        via helper functions from OSMnet or UrbanAccess.
    distance : int
        maximum distance to consider `accessible` (the default is 2000).
    decay : str
        decay type pandana should use "linear", "exp", or "flat"
        (which means no decay). The default is "linear".

    """

    def __init__(self, data, groups, network, w, decay, distance):

        if w and network:
            raise (
                "must pass either a pandana network or a pysal weights object but not both"
            )
        elif network:
            df = calc_access(data,
                             variables=groups,
                             network=network,
                             distance=distance,
                             decay=decay)
        else:
            df = _localize(data, w)
        super().__init__(self, df, groups)
