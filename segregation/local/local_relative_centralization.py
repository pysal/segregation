"""Multigroup Relative Centralization index"""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import libpysal as lps
import numpy as np
from segregation.singlegroup import RelativeCentralization

from .._base import MultiGroupIndex, SpatialExplicitIndex

np.seterr(divide="ignore", invalid="ignore")


def _local_relative_centralization(data, group_pop_var, total_pop_var, W=None, k=5):
    """
    Calculation of Local Relative Centralization index for each unit

    Parameters
    ----------

    data          : a geopandas DataFrame with a geometry column.
    
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
                    
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    
    k_neigh       : integer greater than 0. Default is 5.
                    Number of assumed neighbors for local context (it uses k-nearest algorithm method)
                    
    Returns
    -------

    statistics : np.array(n)
                 Local Relative Centralization values for each unit
                
    core_data  : a pandas DataFrame
                 A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Folch, David C., and Sergio J. Rey. "The centralization index: A measure of local spatial segregation." Papers in Regional Science 95.3 (2016): 555-576.
    
    Reference: :cite:`folch2016centralization`.
    """

    data = data.copy()
    if not W:
        W = lps.weights.KNN.from_dataframe(data, k=5)

    core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    c_lons = data.centroid.map(lambda p: p.x)
    c_lats = data.centroid.map(lambda p: p.y)

    points = list(zip(c_lons, c_lats))
    kd = lps.cg.kdtree.KDTree(np.array(points))

    local_RCEs = np.empty(len(data))

    for i in range(len(data)):

        x = list(W.neighbors.values())[i]
        x.append(
            list(W.neighbors.keys())[i]
        )  # Append in the end the current unit i inside the local context

        local_data = data.iloc[x, :].copy()

        # The center is given by the last position (i.e. the current unit i)
        local_RCE = RelativeCentralization(
            local_data, group_pop_var, total_pop_var, center=len(local_data) - 1
        )

        local_RCEs[i] = local_RCE.statistic

    return local_RCEs, core_data


class LocalRelativeCentralization(MultiGroupIndex, SpatialExplicitIndex):
    """Multigroup Local Simpson's Concentration Index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    groups : list, required
        list of columns on dataframe holding population totals for each group
    w : libpysal.W, optional
        lipysal spatial weights object used to define a local neighborhood. If none is passed,
        a KNN ojbect with k=5 will be used
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
    Based on Folch, David C., and Sergio J. Rey. "The centralization index: A measure of local spatial segregation." Papers in Regional Science 95.3 (2016): 555-576.
    
    Reference: :cite:`folch2016centralization`.
    """

    def __init__(
        self,
        data,
        group_pop_var=None,
        total_pop_var=None,
        w=None,
        network=None,
        distance=None,
        decay=None,
        precompute=None,
        groups=None,
    ):
        """Init."""

        MultiGroupIndex.__init__(self, data, groups)
        if any([w, network, distance]):
            SpatialExplicitIndex.__init__(self)
        aux = _local_relative_centralization(
            self.data, group_pop_var=group_pop_var, total_pop_var=total_pop_var, W=w
        )

        self.statistics = aux[0]
        self.data = aux[1]
        self._function = _local_relative_centralization
