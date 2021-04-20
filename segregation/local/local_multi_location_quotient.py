"""Multigroup dissimilarity index"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np

from .._base import MultiGroupIndex, SpatialImplicitIndex

np.seterr(divide="ignore", invalid="ignore")


def _multi_location_quotient(data, groups):
    """
    Calculation of Location Quotient index for each group and unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings of length k.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n,k)
                 Location Quotient values for each group and unit.
                 Column k has the Location Quotient of position k in groups.
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Isard, Walter. Methods of regional analysis. Vol. 4. Рипол Классик, 1967.
    
    Reference: :cite:`isard1967methods`.

    """
    
    core_data = data[groups]
    
    df = np.array(core_data)
    
    n = df.shape[0]
    K = df.shape[1]
    
    T = df.sum()
    
    ti = df.sum(axis = 1)
    Xk = df.sum(axis = 0)
    
    multi_LQ = (df / np.repeat(ti, K, axis = 0).reshape(n,K)) / (Xk / T)
    
    return multi_LQ, core_data, groups

        
class MultiLocationQuotient(MultiGroupIndex, SpatialImplicitIndex):
    """Multigroup Local Diversity Index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    groups : list, required
        list of columns on dataframe holding population totals for each group
    w : libpysal.weights.KernelW, optional
        lipysal spatial kernel weights object used to define an egohood
    network : pandana.Network
        pandana Network object representing the study area
    distance : int
        Maximum distance (in units of geodataframe CRS) to consider the extent of the egohood
    decay : str
        type of decay function to apply. Options include
    precompute : bool
        Whether to precompute the pandana Network object

    Attributes
    ----------
    statistic : float
        Multigroup Dissimilarity Index value
    core_data : a pandas DataFrame
        DataFrame that contains the columns used to perform the estimate.

    Notes
    -----    
    Reference: :cite:`isard1967methods`.
    """

    def __init__(
        self,
        data,
        groups,
        w=None,
        network=None,
        distance=None,
        decay=None,
        precompute=None,
        function='triangular'
    ):
        """Init."""

        MultiGroupIndex.__init__(self, data, groups)
        if any([w, network, distance]):
            SpatialImplicitIndex.__init__(self, w, network, distance, decay, function, precompute)
        aux = _multi_location_quotient(self.data, self.groups)

        self.statistics = aux[0]
        self.data = aux[1]
        self.groups = aux[2]
        self._function = _multi_location_quotient
