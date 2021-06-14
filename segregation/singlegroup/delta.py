"""Delta Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np

from .._base import SingleGroupIndex, SpatialExplicitIndex


def _delta(data, group_pop_var, total_pop_var):
    """Calculate Delta index.

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
    x = np.array(data[group_pop_var])
    t = np.array(data[total_pop_var])

    if any(t < x):
        raise ValueError(
            "Group of interest population must equal or lower than the total population of the units."
        )

    area = np.array(data.area)

    X = x.sum()
    A = area.sum()

    DEL = 1 / 2 * abs(x / X - area / A).sum()

    core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return DEL, core_data


class Delta(SingleGroupIndex, SpatialExplicitIndex):
    """Delta Index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    group_pop_var : str, required
        name of column on dataframe holding population totals for focal group
    total_pop_var : str, required
        name of column on dataframe holding total overall population

    Attributes
    ----------
    statistic : float
        Delta Index
    core_data : a pandas DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.


    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    Reference: :cite:`massey1988dimensions`.
    """

    def __init__(
        self, data, group_pop_var, total_pop_var, **kwargs,
    ):
        """Init."""
        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        SpatialExplicitIndex.__init__(self,)
        aux = _delta(self.data, self.group_pop_var, self.total_pop_var,)

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _delta
