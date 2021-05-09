"""Absolute Clustering Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, haversine_distances
from libpysal.weights import DistanceBand
from .._base import SingleGroupIndex, SpatialExplicitIndex


def _absolute_clustering(
    data, group_pop_var, total_pop_var, alpha=0.6, beta=0.5, metric="euclidean"
):
    """Calculation of Absolute Clustering index

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

    metric        : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
                    The metric used for the distance between spatial units.
                    If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.

    Returns
    ----------
    statistic : float
                Absolute Clustering Index

    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.
    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.

    Reference: :cite:`massey1988dimensions`.

    """

    if metric not in ["euclidean", "haversine"]:
        raise ValueError("metric must one of 'euclidean', 'haversine'")

    if alpha < 0:
        raise ValueError("alpha must be greater than zero.")

    if beta < 0:
        raise ValueError("beta must be greater than zero.")

    X = data[group_pop_var].values.sum()

    x = data[group_pop_var].values
    t = data[total_pop_var].values
    n = len(data)

    if metric == "euclidean":
        maxd = np.max(
            euclidean_distances([data.centroid.x.values, data.centroid.y.values])
        )
        dist = DistanceBand.from_dataframe(
            data, alpha=-2.0, binary=False, threshold=maxd,
        ).full()[0]

    if metric == "haversine":
        dist = haversine_distances([data.centroid.x.values, data.centroid.y.values]
        )  # This needs to be latitude first!

    np.fill_diagonal(dist, val=((alpha * data.area.values) ** (beta)))

    c = dist.copy()

    ACL = ((((x / X) * (c * x).sum(axis=1)).sum()) - ((X / n ** 2) * c.sum())) / (
        (((x / X) * (c * t).sum(axis=1)).sum()) - ((X / n ** 2) * c.sum())
    )

    core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return ACL, core_data


class AbsoluteClustering(SingleGroupIndex, SpatialExplicitIndex):
    """Absolute Clustering Index.

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
    metric : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
        The metric used for the distance between spatial units.
        If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.


    Attributes
    ----------
    statistic : float
        AbsolutecClustering Index
    core_data : a pandas DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    The pairwise distance between unit i and itself is (alpha * area_of_unit_i) ^ beta.

    Reference: :cite:`massey1988dimensions`.
    """

    def __init__(
        self,
        data,
        group_pop_var,
        total_pop_var,
        alpha=0.6,
        beta=0.5,
        metric="euclidean",
        **kwargs,
    ):
        """Init."""
        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        SpatialExplicitIndex.__init__(self,)
        self.alpha = alpha
        self.beta = beta
        self.metric = metric
        aux = _absolute_clustering(
            self.data,
            self.group_pop_var,
            self.total_pop_var,
            self.alpha,
            self.beta,
            self.metric,
        )

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _absolute_clustering
