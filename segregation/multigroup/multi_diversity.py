"""Multigroup Diversity index"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
from geopandas import GeoDataFrame

from .._base import MultiGroupIndex, SpatialImplicitIndex

np.seterr(divide="ignore", invalid="ignore")


def _multi_diversity(data, groups, normalized=False):
    """Calculate of Multigroup Diveristy Index

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

    Pk = df.sum(axis=0) / df.sum()

    E = -(Pk * np.log(Pk)).sum()

    if normalized:
        K = df.shape[1]
        E = E / np.log(K)
    if isinstance(data, GeoDataFrame):
        core_data = data[[data.geometry.name]].join(core_data)
    return E, core_data, groups


class MultiDiversity(MultiGroupIndex, SpatialImplicitIndex):
    """Multigroup Diversity Index.

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
    normalized : bool. Default is False.
        Whether the resulting index will be divided by its maximum (natural log of the number of groups)
    
    Attributes
    ----------
    statistic : float
        Multigroup Dissimilarity Index value
    core_data : a pandas DataFrame
        DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67 and Theil, Henry. "Statistical decomposition analysis; with applications in the social and administrative sciences". No. 04; HA33, T4.. 1972.

    This is also know as Theil's Entropy Index (Equation 2 of page 37 of Reardon, Sean F., and Glenn Firebaugh. "Measures of multigroup segregation." Sociological methodology 32.1 (2002): 33-67)

    High diversity means less segregation.

    Reference: :cite:`reardon2002measures`.
    """

    def __init__(
        self,
        data,
        groups,
        w=None,
        normalized=False,
        network=None,
        distance=None,
        decay=None,
        precompute=None,
        function='triangular',
        **kwargs
    ):
        """Init."""
        MultiGroupIndex.__init__(self, data, groups)
        self.normalized = normalized
        if any([w, network, distance]):
            SpatialImplicitIndex.__init__(self, w, network, distance, decay, function, precompute)
        aux = _multi_diversity(self.data, self.groups, normalized=self.normalized)

        self.statistic = aux[0]
        self.data = aux[1]
        self.groups = aux[2]
        self._function = _multi_diversity
