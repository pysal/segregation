"""Absolute Centralization Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
from scipy.ndimage import shift

from .._base import SingleGroupIndex, SpatialExplicitIndex


def _absolute_centralization(
    data, group_pop_var, total_pop_var, center="mean", metric="euclidean"
):
    """Calculation of Absolute Centralization index.

    Parameters
    ----------
    data : a geopandas DataFrame with a geometry column.
    group_pop_var : string
        The name of variable in data that contains the population size of the group of interest
    total_pop_var : string
        The name of variable in data that contains the total population of the unit
    center : string, two-dimension values (tuple, list, array) or integer.
        This defines what is considered to be the center of the spatial context under study.
        If string, this can be set to:

            "mean": the center longitude/latitude is the mean of longitudes/latitudes of all units.
            "median": the center longitude/latitude is the median of longitudes/latitudes of all units.
            "population_weighted_mean": the center longitude/latitude is the mean of longitudes/latitudes of all units weighted by the total population.
            "largest_population": the center longitude/latitude is the centroid of the unit with largest total population. If there is a tie in the maximum population, the mean of all coordinates will be taken.

        If tuple, list or array, this argument should be the coordinates of the desired center assuming longitude as first value and latitude second value. Therefore, in the form (longitude, latitude), if tuple, or [longitude, latitude] if list or numpy array.

        If integer, the center will be the centroid of the polygon from data corresponding to the integer interpreted as index.
        For example, if `center = 0` the centroid of the first row of data is used as center, if `center = 1` the second row will be used, and so on.
    metric : string. Can be 'euclidean' or 'haversine'. Default is 'euclidean'.
        The metric used for the distance between spatial units.
        If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.

    Returns
    ----------
    statistic     : float
        Absolute Centralization Index
    core_data     : a geopandas DataFrame
        A geopandas DataFrame that contains the columns used to perform the estimate.
    center_values : list
        The center, in the form [longitude, latitude], values used for the calculation of the centralization distances.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    A discussion of defining the center in this function can be found in https://github.com/pysal/segregation/issues/18.

    Reference: :cite:`massey1988dimensions`.
    """

    if metric not in ["euclidean", "haversine"]:
        raise ValueError("metric must one of 'euclidean', 'haversine'")

    x = np.array(data[group_pop_var])
    t = np.array(data[total_pop_var])

    if any(t < x):
        raise ValueError(
            "Group of interest population must equal or lower than the total population of the units."
        )

    area = np.array(data.area)

    c_lons = np.array(data.centroid.x)
    c_lats = np.array(data.centroid.y)

    if isinstance(center, str):
        if center not in [
            "mean",
            "median",
            "population_weighted_mean",
            "largest_population",
        ]:
            raise ValueError(
                "The center string must one of 'mean', 'median', 'population_weighted_mean', 'largest_population'"
            )

        if center == "mean":
            center_lon = c_lons.mean()
            center_lat = c_lats.mean()

        if center == "median":
            center_lon = np.median(c_lons)
            center_lat = np.median(c_lats)

        if center == "population_weighted_mean":
            center_lon = np.average(c_lons, weights=t)
            center_lat = np.average(c_lats, weights=t)

        if center == "largest_population":
            center_lon = c_lons[np.where(t == t.max())].mean()
            center_lat = c_lats[np.where(t == t.max())].mean()

    if (
        isinstance(center, tuple)
        or isinstance(center, list)
        or isinstance(center, np.ndarray)
    ):
        if np.array(center).shape != (2,):
            raise ValueError("The center tuple/list/array must have length 2.")

        center_lon = center[0]
        center_lat = center[1]

    if isinstance(center, int):
        if (center > len(data) - 1) or (center < 0):
            raise ValueError("The center index must by in the range of data.")

        center_lon = data.iloc[[center]].centroid.x.values[0]
        center_lat = data.iloc[[center]].centroid.y.values[0]

    X = x.sum()
    A = area.sum()

    dlon = c_lons - center_lon
    dlat = c_lats - center_lat

    if metric == "euclidean":
        center_dist = np.sqrt((dlon) ** 2 + (dlat) ** 2)

    if metric == "haversine":
        center_dist = 2 * np.arcsin(
            np.sqrt(
                np.sin(dlat / 2) ** 2
                + np.cos(center_lat) * np.cos(c_lats) * np.sin(dlon / 2) ** 2
            )
        )

    if np.isnan(center_dist).sum() > 0:
        raise ValueError(
            "It not possible to determine the center distance for, at least, one unit. This is probably due to the magnitude of the number of the centroids. We recommend to reproject the geopandas DataFrame."
        )

    asc_ind = center_dist.argsort()

    Xi = np.cumsum(x[asc_ind]) / X
    Ai = np.cumsum(area[asc_ind]) / A

    ACE = np.nansum(shift(Xi, 1, cval=np.NaN) * Ai) - np.nansum(
        Xi * shift(Ai, 1, cval=np.NaN)
    )

    core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    center_values = [center_lon, center_lat]

    return ACE, core_data, center_values


class AbsoluteCentralization(SingleGroupIndex, SpatialExplicitIndex):
    """Absolute Centralization Index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    group_pop_var : str, required
        name of column on dataframe holding population totals for focal group
    total_pop_var : str, required
        name of column on dataframe holding total overall population
    center  : string, two-dimension values (tuple, list, array) or integer.
        This defines what is considered to be the center of the spatial context under study.
        If string, this can be set to:

            "mean": the center longitude/latitude is the mean of longitudes/latitudes of all units.
            "median": the center longitude/latitude is the median of longitudes/latitudes of all units.
            "population_weighted_mean": the center longitude/latitude is the mean of longitudes/latitudes of all units weighted by the total population.
            "largest_population": the center longitude/latitude is the centroid of the unit with largest total population. If there is a tie in the maximum population, the mean of all coordinates will be taken.
    metric : str
        The metric used for the distance between spatial units.
        If the projection of the CRS of the geopandas DataFrame field is in degrees, this should be set to 'haversine'.


    Attributes
    ----------
    statistic : float
        AbsoluteCentralization Index
    core_data : a pandas DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    A discussion of defining the center in this function can be found in https://github.com/pysal/segregation/issues/18.

    Reference: :cite:`massey1988dimensions`.
    """

    def __init__(
        self,
        data,
        group_pop_var,
        total_pop_var,
        center="mean",
        metric="euclidean",
        **kwargs,
    ):
        """Init."""
        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        SpatialExplicitIndex.__init__(self,)
        self.center = center
        self.metric = metric
        aux = _absolute_centralization(
            self.data, self.group_pop_var, self.total_pop_var, self.center, self.metric,
        )

        self.statistic = aux[0]
        self.core_data = aux[1]
        self.center_values = aux[2]
        self._function = _absolute_centralization
