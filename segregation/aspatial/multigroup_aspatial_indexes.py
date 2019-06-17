"""
Multigroup Aspatial based Segregation Metrics
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"



import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances

__all__ = ['Multi_Dissim',
           'Multi_Gini_Seg',
           'Multi_Normalized_Exposure',
           'Multi_Information_Theory',
           'Multi_Relative_Diversity',
           'Multi_Squared_Coefficient_of_Variation',
           'Multi_Diversity',
           'Simpsons_Concentration',
           'Simpsons_Interaction',
           'Multi_Divergence']

def _multi_dissim(data, groups):
    """
    Calculation of Multigroup Dissimilarity index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistic : float
                Multigroup Dissimilarity Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Sakoda, James M. "A generalized index of dissimilarity." Demography 18.2 (1981): 245-250.
    
    Reference: :cite:`sakoda1981generalized`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    n = df.shape[0]
    K = df.shape[1]
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    Pk = df.sum(axis = 0) / df.sum()
    
    Is = (Pk * (1 - Pk)).sum()
    
    multi_D = 1/(2 * T * Is) * np.multiply(abs(pik - Pk), np.repeat(ti, K, axis=0).reshape(n,K)).sum()
    
    return multi_D, core_data


class Multi_Dissim:
    """
    Calculation of Multigroup Dissimilarity index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic : float
                Multigroup Dissimilarity Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from segregation.multigroup_aspatial import Multi_Dissim
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = Multi_Dissim(input_df, groups_list)
    >>> index.statistic
    0.41340872573177806

    Notes
    -----
    Based on Sakoda, James M. "A generalized index of dissimilarity." Demography 18.2 (1981): 245-250.
    
    Reference: :cite:`sakoda1981generalized`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_dissim(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_dissim
        
        
        
def _multi_gini_seg(data, groups):
    """
    Calculation of Multigroup Gini Segregation index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistic : float
                Multigroup Gini Segregation Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Reference: :cite:`reardon2002measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    K = df.shape[1]
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    Pk = df.sum(axis = 0) / df.sum()
    Is = (Pk * (1 - Pk)).sum()
    
    elements_sum = np.empty(K)
    for k in range(K):
        aux = np.multiply(np.outer(ti, ti), manhattan_distances(pik[:,k].reshape(-1, 1))).sum()
        elements_sum[k] = aux
        
    multi_Gini_Seg = elements_sum.sum() / (2 * (T ** 2) * Is)
    
    return multi_Gini_Seg, core_data


class Multi_Gini_Seg:
    """
    Calculation of Multigroup Gini Segregation index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic : float
                Multigroup Gini Segregation Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from segregation.multigroup_aspatial import Multi_Gini_Seg
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = Multi_Gini_Seg(input_df, groups_list)
    >>> index.statistic
    0.5456349992598081

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Reference: :cite:`reardon2002measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_gini_seg(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_gini_seg
        


def _multi_normalized_exposure(data, groups):
    """
    Calculation of Multigroup Normalized Exposure index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistic : float
                Multigroup Normalized Exposure Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Reference: :cite:`reardon2002measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    Pk = df.sum(axis = 0) / df.sum()
    
    MNE = ((ti[:,None] * (pik - Pk) ** 2) / (1 - Pk)).sum() / T
    
    return MNE, core_data


class Multi_Normalized_Exposure:
    """
    Calculation of Multigroup Normalized Exposure index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic : float
                Multigroup Normalized Exposure Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from segregation.multigroup_aspatial import Multi_Normalized_Exposure
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = Multi_Normalized_Exposure(input_df, groups_list)
    >>> index.statistic
    0.18821879029994157

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Reference: :cite:`reardon2002measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_normalized_exposure(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_normalized_exposure
        
        
        
def _multi_information_theory(data, groups):
    """
    Calculation of Multigroup Information Theory index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistic : float
                Multigroup Information Theory Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Reference: :cite:`reardon2002measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    Pk = df.sum(axis = 0) / df.sum()
    
    # The natural logarithm is used, but this could be used with any base following Footnote 3 of pg. 37
    # of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    E = (Pk * np.log(1/Pk)).sum()
    
    MIT = np.nansum(ti[:,None] * pik * np.log(pik / Pk)) / (T*E)
    
    return MIT, core_data


class Multi_Information_Theory:
    """
    Calculation of Multigroup Information Theory index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic : float
                Multigroup Information Theory Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from segregation.multigroup_aspatial import Multi_Information_Theory
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = Multi_Information_Theory(input_df, groups_list)
    >>> index.statistic
    0.1710160297858887

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Reference: :cite:`reardon2002measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_information_theory(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_information_theory
        
        
        
def _multi_relative_diversity(data, groups):
    """
    Calculation of Multigroup Relative Diversity index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistic : float
                Multigroup Relative Diversity Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Reardon, Sean F. "Measures of racial diversity and segregation in multigroup and hierarchically structured populations." annual meeting of the Eastern Sociological Society, Philadelphia, PA. 1998.

    High diversity means less segregation.
    
    Reference: :cite:`reardon1998measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    Pk = df.sum(axis = 0) / df.sum()
    Is = (Pk * (1 - Pk)).sum()
    
    MRD = (ti[:,None] * (pik - Pk) ** 2).sum() / (T * Is)
    
    return MRD, core_data





class Multi_Relative_Diversity:
    """
    Calculation of Multigroup Relative Diversity index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic : float
                Multigroup Relative Diversity Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from segregation.multigroup_aspatial import Multi_Relative_Diversity
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = Multi_Relative_Diversity(input_df, groups_list)
    >>> index.statistic
    0.15820019878220337

    Notes
    -----
    Based on Reardon, Sean F. "Measures of racial diversity and segregation in multigroup and hierarchically structured populations." annual meeting of the Eastern Sociological Society, Philadelphia, PA. 1998.

    High diversity means less segregation.
    
    Reference: :cite:`reardon1998measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_relative_diversity(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_relative_diversity
        
        
        
def _multi_squared_coefficient_of_variation(data, groups):
    """
    Calculation of Multigroup Squared Coefficient of Variation index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistic : float
                Multigroup Squared Coefficient of Variation Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.

    Reference: :cite:`reardon2002measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    K = df.shape[1]
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    Pk = df.sum(axis = 0) / df.sum()
    
    C = ((ti[:,None] * (pik - Pk) ** 2) / (T * (K - 1) * Pk)).sum()
    
    return C, core_data




class Multi_Squared_Coefficient_of_Variation:
    """
    Calculation of Multigroup Squared Coefficient of Variation index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic : float
                Multigroup Squared Coefficient of Variation Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from segregation.multigroup_aspatial import Multi_Squared_Coefficient_of_Variation
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = Multi_Squared_Coefficient_of_Variation(input_df, groups_list)
    >>> index.statistic
    0.11875484641127525
    
    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Reference: :cite:`reardon2002measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_squared_coefficient_of_variation(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_squared_coefficient_of_variation
        
        
def _multi_diversity(data, groups, normalized = False):
    """
    Calculation of Multigroup Diversity index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistic  : float
                 Multigroup Diversity Index
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.
                
    normalized : bool. Default is False.
                 Wheter the resulting index will be divided by its maximum (natural log of the number of groups)

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67 and Theil, Henry. "Statistical decomposition analysis; with applications in the social and administrative sciences". No. 04; HA33, T4.. 1972.
    
    This is also know as Theil's Entropy Index (Equation 2 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67)
    
    High diversity means less segregation.
    
    Reference: :cite:`reardon2002measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    Pk = df.sum(axis = 0) / df.sum()
    
    E = - (Pk * np.log(Pk)).sum()
    
    if normalized:
        K = df.shape[1]
        E = E / np.log(K)
    
    return E, core_data


class Multi_Diversity:
    """
    Calculation of Multigroup Diversity index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic : float
                Multigroup Diversity Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from segregation.multigroup_aspatial import Multi_Diversity
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = Multi_Diversity(input_df, groups_list)
    >>> index.statistic
    0.9733112243997906
    
    You can also fit the normalized version of the multigroup diversity index.
    
    >>> normalized_index = Multi_Diversity(input_df, groups_list, normalized = True)
    >>> normalized_index.statistic
    0.7020956383415715

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67 and Theil, Henry. "Statistical decomposition analysis; with applications in the social and administrative sciences". No. 04; HA33, T4.. 1972.
    
    This is also know as Theil's Entropy Index (Equation 2 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67)
    
    High diversity means less segregation.
    
    Reference: :cite:`reardon2002measures`.

    """
    
    def __init__(self, data, groups, normalized = False):
        
        aux = _multi_diversity(data, groups, normalized)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_diversity
        
        

def _simpsons_concentration(data, groups):
    """
    Calculation of Simpson's Concentration index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistic  : float
                 Simpson's Concentration Index
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Simpson, Edward H. "Measurement of diversity." nature 163.4148 (1949): 688.
    
    Simpson's concentration index (Lambda) can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to belong to the same group.

    Higher values means higher segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`simpson1949measurement`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    Pk = df.sum(axis = 0) / df.sum()
    
    Lambda = (Pk * Pk).sum()
    
    return Lambda, core_data



class Simpsons_Concentration:
    """
    Calculation of Simpson's Concentration index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic  : float
                 Simpson's Concentration Index
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from segregation.multigroup_aspatial import Simpsons_Concentration
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = Simpsons_Concentration(input_df, groups_list)
    >>> index.statistic
    0.49182413151957904
    
    Notes
    -----
    Based on Simpson, Edward H. "Measurement of diversity." nature 163.4148 (1949): 688.
    
    Simpson's concentration index (Lambda) can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to belong to the same group.

    Higher values means higher segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`simpson1949measurement`.

    """
    
    def __init__(self, data, groups):
        
        aux = _simpsons_concentration(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _simpsons_concentration
        
        
        
def _simpsons_interaction(data, groups):
    """
    Calculation of Simpson's Interaction index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistic  : float
                 Simpson's Interaction Index
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Equation 1 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Simpson's interaction index (I) can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to not belong to the same group.

    Higher values means lesser segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`reardon2002measures`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    Pk = df.sum(axis = 0) / df.sum()
    
    I = (Pk * (1 - Pk)).sum()
    
    return I, core_data



class Simpsons_Interaction:
    """
    Calculation of Simpson's Interaction index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic  : float
                 Simpson's Interaction Index
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from segregation.multigroup_aspatial import Simpsons_Interaction
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = Simpsons_Interaction(input_df, groups_list)
    >>> index.statistic
    0.508175868480421
    
    Notes
    -----
    Based on Equation 1 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Simpson's interaction index (I) can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to not belong to the same group.

    Higher values means lesser segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`reardon2002measures`.

    """
    
    def __init__(self, data, groups):
        
        aux = _simpsons_interaction(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _simpsons_interaction
        
        
        
def _multi_divergence(data, groups):
    """
    Calculation of Multigroup Divergence index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistic : float
                Multigroup Divergence Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Roberto, Elizabeth. "The Divergence Index: A Decomposable Measure of Segregation and Inequality." arXiv preprint arXiv:1508.01167 (2015).
    
    Reference: :cite:`roberto2015divergence`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    pik = df/ti[:,None]
    Pk = df.sum(axis = 0) / df.sum()
    
    Di = np.nansum(pik * np.log(pik / Pk), axis = 1)

    Divergence_Index = ((ti / T) * Di).sum()
    
    return Divergence_Index, core_data


class Multi_Divergence:
    """
    Calculation of Multigroup Divergence index

    Parameters
    ----------

    data   : a pandas DataFrame
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Attributes
    ----------

    statistic : float
                Multigroup Divergence Index
                
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Examples
    --------
    In this example, we are going to use 2000 Census Tract Data for Sacramento MSA, CA. The groups of interest are White, Black, Asian and Hispanic population.
    
    Firstly, we need to perform some import the modules and the respective function.
    
    >>> import libpysal
    >>> import geopandas as gpd
    >>> from segregation.multigroup_aspatial import Multi_Divergence
    
    Then, we read the data and create an auxiliary list with only the necessary columns for fitting the index.
    
    >>> input_df = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
    >>> groups_list = ['WHITE_', 'BLACK_', 'ASIAN_','HISP_']
    
    The value is estimated below.
    
    >>> index = Multi_Divergence(input_df, groups_list)
    >>> index.statistic
    0.16645182134289443

    Notes
    -----
    Based on Roberto, Elizabeth. "The Divergence Index: A Decomposable Measure of Segregation and Inequality." arXiv preprint arXiv:1508.01167 (2015).
    
    Reference: :cite:`roberto2015divergence`.

    """
    
    def __init__(self, data, groups):
        
        aux = _multi_divergence(data, groups)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _multi_divergence