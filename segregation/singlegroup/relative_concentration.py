"""Relative Concentration Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np

from .._base import SingleGroupIndex, SpatialExplicitIndex


def _relative_concentration(data, group_pop_var, total_pop_var):
    """Calculate Relative Concentration index.

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
    x = np.array(data[group_pop_var])
    t = np.array(data[total_pop_var])

    if any(t < x):
        raise ValueError(
            "Group of interest population must equal or lower than the total population of the units."
        )

    area = np.array(data.area)

    y = t - x

    X = x.sum()
    Y = y.sum()
    T = t.sum()

    # Create the indexes according to the area ordering
    des_ind = (-area).argsort()
    asc_ind = area.argsort()

    # A discussion about the extraction of n1 and n2 can be found in https://github.com/pysal/segregation/issues/43
    n1 = np.where(((np.cumsum(t[asc_ind]) / T) < X / T) == False)[0][0] + 1
    n2_aux = np.where(((np.cumsum(t[des_ind]) / T) < X / T) == False)[0][0] + 1
    n2 = len(data) - n2_aux

    n = data.shape[0]
    T1 = t[asc_ind][0:n1].sum()
    T2 = t[asc_ind][n2:n].sum()

    RCO = (
        (
            ((x[asc_ind] * area[asc_ind] / X).sum())
            / ((y[asc_ind] * area[asc_ind] / Y).sum())
        )
        - 1
    ) / (
        (
            ((t[asc_ind] * area[asc_ind])[0:n1].sum() / T1)
            / ((t[asc_ind] * area[asc_ind])[n2:n].sum() / T2)
        )
        - 1
    )

    core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return RCO, core_data


class RelativeConcentration(SingleGroupIndex, SpatialExplicitIndex):
    """Relative Concentration Index.

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
        Relative Conrentration Index
    core_data : a pandas DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.

    Reference: :cite:`massey1988dimensions`.
    """

    def __init__(
        self, data, group_pop_var, total_pop_var, **kwargs,
    ):
        """Init."""
        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        SpatialExplicitIndex.__init__(self,)
        aux = _relative_concentration(
            self.data, self.group_pop_var, self.total_pop_var,
        )

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _relative_concentration
