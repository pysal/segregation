"""ConProf Segregation Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import geopandas as gpd
import numpy as np
import pandas as pd

from .._base import SingleGroupIndex, SpatialImplicitIndex


def _conprof(data, group_pop_var, total_pop_var, m=1000):
    """Calculation of Concentration Profile.

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
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.

    Reference: :cite:`hong2014measuring`.

    """
    if type(m) is not int:
        raise TypeError("m must be a string.")

    if m < 2:
        raise ValueError("m must be greater than 1.")

    x = np.array(data[group_pop_var])
    t = np.array(data[total_pop_var])

    if any(t < x):
        raise ValueError(
            "Group of interest population must equal or lower than the total population of the units."
        )

    def calculate_vt(th):
        g_t_i = np.where(x / t >= th, 1, 0)
        v_t = (g_t_i * x).sum() / x.sum()
        return v_t

    grid = np.linspace(0, 1, m)
    curve = np.array(list(map(calculate_vt, grid)))

    threshold = x.sum() / t.sum()
    R = (
        threshold
        - ((curve[grid < threshold]).sum() / m - (curve[grid >= threshold]).sum() / m)
    ) / (1 - threshold)

    return R, grid, curve, data


class ConProf(SingleGroupIndex, SpatialImplicitIndex):
    """ConProf Index.

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
                ConProf Index
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.

    Reference: :cite:`hong2014measuring`.
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
        aux = _conprof(self.data, self.group_pop_var, self.total_pop_var)

        self.statistic = aux[0]
        self.grid = aux[1]
        self.curve = aux[2]
        self.core_data = aux[3]
        self._function = _conprof

    def plot(self):
        """Plot the Concentration Profile."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("plotting requires `matplotlib`")
        graph = plt.scatter(self.grid, self.curve, s=0.1)
        return graph
