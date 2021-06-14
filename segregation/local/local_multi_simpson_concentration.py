"""Multigroup dissimilarity index"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np

from .._base import MultiGroupIndex, SpatialImplicitIndex

np.seterr(divide="ignore", invalid="ignore")


def _multi_local_simpson_concentration(data, groups):
    """
    Calculation of Local Simpson concentration index for each unit

    Parameters
    ----------

    data   : a pandas DataFrame of n rows
    
    groups : list of strings.
             The variables names in data of the groups of interest of the analysis.

    Returns
    -------

    statistics : np.array(n)
                 Local Simpson concentration values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on the local version of Equation 1 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67.
    
    Simpson's concentration index can be simply interpreted as the probability that two individuals chosen at random and independently from the population will be found to belong to the same group.

    Higher values means lesser segregation.
    
    Simpson's Concentration + Simpson's Interaction = 1
    
    Reference: :cite:`reardon2002measures`.

    """

    core_data = data[groups]

    df = np.array(core_data)

    ti = df.sum(axis=1)
    pik = df / ti[:, None]

    local_SC = np.nansum(pik * pik, axis=1)

    return local_SC, core_data, groups


class MultiLocalSimpsonConcentration(MultiGroupIndex, SpatialImplicitIndex):
    """Multigroup Local Simpson's Concentration Index.

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
    Based on Eq. 6 of pg. 139 (individual unit case) of Reardon, Sean F., and David O’Sullivan. "Measures of spatial segregation." Sociological methodology 34.1 (2004): 121-162.
    
    Reference: :cite:`reardon2004measures`.
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
        aux = _multi_local_simpson_concentration(self.data, self.groups)

        self.statistics = aux[0]
        self.data = aux[1]
        self.groups = aux[2]
        self._function = _multi_local_simpson_concentration
