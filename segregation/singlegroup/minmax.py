"""MinMax Segregation Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import geopandas as gpd
import numpy as np

from .._base import SingleGroupIndex, SpatialImplicitIndex


def _min_max(data, group_pop_var, total_pop_var):
    """MinMax Segregation index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame
        Dataframe or geodataframe if spatial index holding data for location of interest
    group_pop_var : string
        Variable containing the population count of the group of interest
    total_pop_var : string
        Variable in data that contains the total population count of the unit

    Returns
    ----------
    statistic : float
        MinMax index statistic value
    core_data : pandas.DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on O'Sullivan & Wong (2007). A Surface‐Based Approach to Measuring Spatial Segregation.
    Geographical Analysis 39 (2). https://doi.org/10.1111/j.1538-4632.2007.00699.x

    Reference: :cite:`osullivanwong2007surface`.

    We'd like to thank @AnttiHaerkoenen for this contribution!

    """
    data["group_1_pop_var_norm"] = data[group_pop_var] / data[group_pop_var].sum()
    data["group_2_pop_var_norm"] = (
        data["group_2_pop_var"] / data["group_2_pop_var"].sum()
    )

    density_1 = data["group_1_pop_var_norm"].values
    density_2 = data["group_2_pop_var_norm"].values
    densities = np.vstack([density_1, density_2])
    v_union = densities.max(axis=0).sum()
    v_intersect = densities.min(axis=0).sum()

    MM = 1 - v_intersect / v_union

    if not isinstance(data, gpd.GeoDataFrame):
        data = data[[group_pop_var, total_pop_var]]

    else:
        data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return MM, data


class MinMax(SingleGroupIndex, SpatialImplicitIndex):
    """Min-Max Index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    group_pop_var : str, required
        name of column on dataframe holding population totals for focal group
    total_pop_var : str, required
        name of column on dataframe holding total overall population
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
                MinMax Index
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on O'Sullivan & Wong (2007). A Surface‐Based Approach to Measuring Spatial Segregation.
    Geographical Analysis 39 (2). https://doi.org/10.1111/j.1538-4632.2007.00699.x

    Reference: :cite:`osullivanwong2007surface`.

    We'd like to thank @AnttiHaerkoenen for this contribution!
    """

    def __init__(
        self,
        data,
        group_pop_var,
        total_pop_var,
        w=None,
        network=None,
        distance=None,
        decay=None,
        function="triangular",
        precompute=None,
        **kwargs
    ):
        """Init."""
        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        if any([w, network, distance]):
            SpatialImplicitIndex.__init__(
                self, w, network, distance, decay, function, precompute
            )
        aux = _min_max(self.data, self.group_pop_var, self.total_pop_var)

        self.statistic = aux[0]
        self.data = aux[1]
        self._function = _min_max
