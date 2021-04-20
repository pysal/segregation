"""Relative Clustering Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, haversine_distances

from .._base import SingleGroupIndex, SpatialExplicitIndex


def _relative_clustering(
    data, group_pop_var, total_pop_var, alpha=0.6, beta=0.5, metric="euclidean"
):
    """
    Calculation of Relative Clustering index

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
                Relative Clustering Index

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

    data = data.assign(
        xi=data[group_pop_var], yi=data[total_pop_var] - data[group_pop_var]
    )

    X = data.xi.sum()
    Y = data.yi.sum()

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    if metric == "euclidean":
        dist = euclidean_distances(pd.DataFrame({"c_lats": c_lats, "c_lons": c_lons}))

    if metric == "haversine":
        dist = haversine_distances(
            pd.DataFrame({"c_lats": c_lats, "c_lons": c_lons})
        )  # This needs to be latitude first!

    c = np.exp(-dist)

    if c.sum() < 10 ** (-15):
        raise ValueError(
            "It not possible to determine accurately the exponential of the negative distances. This is probably due to the large magnitude of the centroids numbers. It is recommended to reproject the geopandas DataFrame. Also, if this is a not lat-long CRS, it is recommended to set metric to 'haversine'"
        )

    np.fill_diagonal(c, val=np.exp(-((alpha * data.area) ** (beta))))

    Pxx = ((np.array(data.xi) * c).T * np.array(data.xi)).sum() / X ** 2
    Pyy = ((np.array(data.yi) * c).T * np.array(data.yi)).sum() / Y ** 2
    RCL = Pxx / Pyy - 1

    if np.isnan(RCL):
        raise ValueError(
            "It not possible to determine the distance between, at least, one pair of units. This is probably due to the magnitude of the number of the centroids. We recommend to reproject the geopandas DataFrame."
        )

    core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return RCL, core_data


class RelativeClustering(SingleGroupIndex, SpatialExplicitIndex):
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
    metric : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
        The metric used for the distance between spatial units.
        If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.


    Attributes
    ----------
    statistic : float
        SpatialDissim Index
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
        aux = _relative_clustering(
            self.data,
            self.group_pop_var,
            self.total_pop_var,
            self.alpha,
            self.beta,
            self.metric,
        )

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _relative_clustering