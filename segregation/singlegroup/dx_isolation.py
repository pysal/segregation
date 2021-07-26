"""Distance Decay Isolation Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
import pandas as pd
from ..util import generate_distance_matrix

from .._base import SingleGroupIndex, SpatialExplicitIndex


def _distance_decay_isolation(data, group_pop_var, total_pop_var, alpha=0.6, beta=0.5):
    """Calculate of Distance Decay Isolation index.

    Parameters
    ----------
    data          : a geopandas DataFrame with a geometry column.
    group_pop_var : string
                    The name of variable in data that contains the population size of the group of interest
    total_pop_var : string
                    The name of variable in data that contains the total population of the unit
    alpha         : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    beta          : float
                    A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5

    Returns
    ----------
    statistic : float
                Distance Decay Isolation Index
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    It may be interpreted as the probability that the next person a group member meets anywhere in space is from the same group.

    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.

    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.

    Reference: :cite:`morgan1983distance`.

    """
    if alpha < 0:
        raise ValueError("alpha must be greater than zero.")

    if beta < 0:
        raise ValueError("beta must be greater than zero.")

    x = np.array(data[group_pop_var])
    t = np.array(data[total_pop_var])

    if any(t < x):
        raise ValueError(
            "Group of interest population must equal or lower than the total population of the units."
        )

    X = x.sum()

    dist = generate_distance_matrix(data)

    np.fill_diagonal(dist, val=np.exp(-((alpha * data.area.values) ** (beta))))

    c = 1 - dist.copy()  # proximity matrix

    Pij = np.multiply(c, t) / np.sum(np.multiply(c, t), axis=1)

    DDxPx = (
        np.array(x / X) * np.nansum(np.multiply(Pij, np.array(x / t)), axis=1)
    ).sum()

    core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return DDxPx, core_data


class DistanceDecayIsolation(SingleGroupIndex, SpatialExplicitIndex):
    """Distance-Decay Isolation Index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    group_pop_var : str, required
        name of column on dataframe holding population totals for focal group
    total_pop_var : str, required
        name of column on dataframe holding total overall population
    alpha  : float
        A parameter that estimates the extent of the proximity within the same unit. Default value is 0.6
    beta : float
        A parameter that estimates the extent of the proximity within the same unit. Default value is 0.5

    Attributes
    ----------
    statistic : float
        Distance Decay Isolation Index
    core_data : a pandas DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    It may be interpreted as the probability that the next person a group member meets anywhere in space is from the same group.

    Based on Morgan, Barrie S. "A distance-decay based interaction index to measure residential segregation." Area (1983): 211-217.

    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.

    Reference: :cite:`morgan1983distance`.
    """

    def __init__(
        self, data, group_pop_var, total_pop_var, alpha=0.6, beta=0.5, **kwargs,
    ):
        """Init."""
        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        SpatialExplicitIndex.__init__(self,)
        self.alpha = alpha
        self.beta = beta
        aux = _distance_decay_isolation(
            self.data, self.group_pop_var, self.total_pop_var, self.alpha, self.beta,
        )

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _distance_decay_isolation
