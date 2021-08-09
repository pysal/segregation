"""Spatial Proximity Profile Segregation Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
from libpysal.weights import Queen
from scipy.sparse.csgraph import floyd_warshall
from numba import njit
from .._base import SingleGroupIndex, SpatialExplicitIndex


def _spatial_prox_profile(data, group_pop_var, total_pop_var, w, m):
    """Calculate Spatial Proximity Profile.

    Parameters
    ----------
    data : geopandas.GeoDataFrame (required)
        GeoDataFrame with valid geometry column and columns for group population and total population
    group_pop_var : string
        The name of variable in data that contains the population size of the group of interest
    total_pop_var : string
        The name of variable in data that contains the total population of the unit
    w: libpysal.weights.W
        pysal spatial weights object measuring connectivity between geographic units. If nNne, a Queen object will be created
    m : int
        a numeric value indicating the number of thresholds to be used. Default value is 1000.
        A large value of m creates a smoother-looking graph and a more precise spatial proximity profile value but slows down the calculation speed.

    Returns
    ----------
    statistic : float
        Spatial Proximity Index
    core_data : geopandas.GeoDataFrame
        A GeoDataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.

    Reference: :cite:`hong2014measuring`.

    """
    # Create the shortest distance path between two pair of units using Shimbel matrix.
    # This step was well discussed in https://github.com/pysal/segregation/issues/5.
    if not w:
        w = Queen.from_dataframe(data)
    delta = floyd_warshall(csgraph=w.sparse, directed=False)
    group_vals = data[group_pop_var].to_numpy()
    total_vals = data[total_pop_var].to_numpy()
    
    grid = np.linspace(0, 1, m)

    @njit(fastmath=True, error_model="numpy")
    def calc(grid):
        def calculate_etat(t):
            g_t_i = np.where(np.divide(group_vals, total_vals) >= t, True, False)
            k = g_t_i.sum()

            # i and j only varies in the units subset within the threshold in eta_t of Hong (2014).
            sub_delta_ij = delta[g_t_i, :][:, g_t_i]

            den = sub_delta_ij.sum()
            eta_t = (k ** 2 - k) / den
            return eta_t

        results = np.empty(len(grid))
        for i, est in enumerate(grid):
            aux = calculate_etat(est)
            results[i] = aux
        return results

    aux = calc(grid)
    aux[aux == np.inf] = 0
    aux[aux == -np.inf] = 0
    curve = np.nan_to_num(aux, 0)

    threshold = data[group_pop_var].sum() / data[total_pop_var].sum()
    SPP = (
        threshold
        - ((curve[grid < threshold]).sum() / m - (curve[grid >= threshold]).sum() / m)
    ) / (1 - threshold)

    core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return SPP, grid, curve, core_data


class SpatialProxProf(SingleGroupIndex, SpatialExplicitIndex):
    """Spatial Proximity Profile Index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    group_pop_var : str, required
        name of column on dataframe holding population totals for focal group
    total_pop_var : str, required
        name of column on dataframe holding total overall population
    w: libpysal.weights.W
        pysal spatial weights object measuring connectivity between geographic units. If nNne, a Queen object will be created
    m : int
        a numeric value indicating the number of thresholds to be used. Default value is 1000.
        A large value of m creates a smoother-looking graph and a more precise spatial proximity
        profile value but slows down the calculation speed.

    Attributes
    ----------
    statistic : float
        Spatial Prox Profile Index
    core_data : a pandas DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Hong, Seong-Yun, and Yukio Sadahiro. "Measuring geographic segregation: a graph-based approach." Journal of Geographical Systems 16.2 (2014): 211-231.

    Reference: :cite:`hong2014measuring`.
    """

    def __init__(self, data, group_pop_var, total_pop_var, w=None, m=1000, **kwargs):
        """Init."""
        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        SpatialExplicitIndex.__init__(self,)
        self.m = m
        self.w = w
        aux = _spatial_prox_profile(
            self.data, self.group_pop_var, self.total_pop_var, self.w, self.m
        )

        self.statistic = aux[0]
        self.grid = aux[1]
        self.curve = aux[2]
        self.core_data = aux[3]
        self._function = _spatial_prox_profile

    def plot(self):
        """Plot the Spatial Proximity Profile."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("This method relies on importing `matplotlib`")
        graph = plt.scatter(self.grid, self.curve, s=0.1)
        return graph
