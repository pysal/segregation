"""Density-Corrected Dissim Segregation Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .._base import SingleGroupIndex, SpatialImplicitIndex


# Constructing function that returns $n(\hat{\theta}_j)$
def _return_optimal_theta(theta_j):
    def fold_norm(x):

        y = (-1) * (norm.pdf(x - theta_j) + norm.pdf(x + theta_j))
        return y

    initial_guesses = np.array(0)
    res = minimize(
        fold_norm, initial_guesses, method="nelder-mead", options={"xatol": 1e-5}
    )
    return res.final_simplex[0][1][0]


def _density_corrected_dissim(
    data,
    group_pop_var,
    total_pop_var,
):
    """Calculate Density Corrected Dissimilarity index.

    Parameters
    ----------
    data :  pandas.DataFrame
        DataFrame storing necessary data
    group_pop_var : string
        The name of variable in data that contains the population size of the group of interest
    total_pop_var : string
        The name of variable in data that contains the total population of the unit
    xtol : float
        The degree of tolerance in the optimization process of returning optimal theta_j

    Returns
    ----------
    statistic : float
        Dissimilarity with Density-Correction (density correction from Allen, Rebecca et al. (2015))
    core_data : pandas.DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Allen, Rebecca, et al. "More reliable inference for the dissimilarity index of segregation." The econometrics journal 18.1 (2015): 40-66.

    Reference: :cite:`allen2015more`.
    """
    g = np.array(data[group_pop_var])
    t = np.array(data[total_pop_var])

    other_group_pop = t - g

    # Group 0: minority group
    p0_i = g / g.sum()
    n0 = g.sum()

    # Group 1: complement group
    p1_i = other_group_pop / other_group_pop.sum()
    n1 = other_group_pop.sum()

    sigma_hat_j = np.sqrt(((p1_i * (1 - p1_i)) / n1) + ((p0_i * (1 - p0_i)) / n0))
    theta_hat_j = abs(p1_i - p0_i) / sigma_hat_j

    optimal_thetas = pd.Series(data=theta_hat_j).apply(_return_optimal_theta)

    Ddc = np.multiply(sigma_hat_j, optimal_thetas).sum() / 2

    if not isinstance(data, gpd.GeoDataFrame):
        core_data = data[[group_pop_var, total_pop_var]]

    else:
        core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return Ddc, core_data


class DensityCorrectedDissim(SingleGroupIndex, SpatialImplicitIndex):
    """Density Corrected Dissimilarity Index.

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
        Segregation Index
    core_data : a pandas DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Allen, Rebecca, et al. "More reliable inference for the dissimilarity index of segregation." The econometrics journal 18.1 (2015): 40-66.

    Reference: :cite:`allen2015more`.
    """

    def __init__(
        self,
        data,
        group_pop_var,
        total_pop_var,
        w=None,
        network=None,
        distance=None,
        decay="linear",
        precompute=None,
        function="triangular",
        **kwargs
    ):
        """Init."""

        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        if any([w, network, distance]):
            SpatialImplicitIndex.__init__(
                self, w, network, distance, decay, function, precompute
            )
        aux = _density_corrected_dissim(
            self.data, self.group_pop_var, self.total_pop_var
        )

        self.statistic = aux[0]
        self.data = aux[1]
        self._function = _density_corrected_dissim
