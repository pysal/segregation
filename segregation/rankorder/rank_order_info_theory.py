"""Rank-Order Information Theory Index."""

import numpy as np

from .._base import RankOrderIndex, SpatialImplicitIndex
from ..singlegroup import Entropy
from ._utils import _core_data, _rank_order, deltas_info_theory, entropy_weight


def _rank_order_info_theory(data, groups, degree=4):
    """Calculate the Rank-Order Information Theory Index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame
        Dataframe or geodataframe if spatial index holding data for location of interest
    groups : list
        Columns on `data` holding population counts for each category of an
        ordered variable, listed from the lowest category to the highest
    degree : int
        Degree of the polynomial fit through the segregation profile

    Returns
    ----------
    statistic : float
        Rank-order information theory index statistic value
    core_data : pandas.DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.
    groups : list
        The ordered category columns used to perform the estimate.
    profile : pandas.DataFrame
        The segregation profile: the pairwise index at each usable threshold.
    deltas : numpy.ndarray
        Reardon's delta coefficients for the fitted polynomial degree.
    model : statsmodels results
        The fitted weighted least squares model.

    Notes
    -----
    Reardon's :math:`H^R`, the information theory index integrated across the
    thresholds of an ordered variable using the entropy weight
    :math:`E(p) = -2[p \\ln p + (1 - p)\\ln(1 - p)]`, which is moderately
    sensitive to the extremes of the distribution.

    Based on Reardon, Sean F. "Measures of income segregation."
    Working paper, Stanford Center for Education Policy Analysis (2011).

    Reference: :cite:`reardon2011measures`.
    """
    statistic, profile, deltas, model = _rank_order(
        data, groups, degree, Entropy, entropy_weight, deltas_info_theory
    )
    return statistic, _core_data(data, groups), groups, profile, deltas, model


class RankOrderInfoTheory(RankOrderIndex, SpatialImplicitIndex):
    """Rank-Order Information Theory Index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    groups : list, required
        columns on dataframe holding population counts for each category of an
        ordered variable, listed from the lowest category to the highest. The
        categories must partition the population; the total for each unit is
        the sum across these columns.
    degree : int
        degree of the polynomial fit through the segregation profile, by default 4
    w : libpysal.weights.KernelW, optional
        lipysal spatial kernel weights object used to define an egohood
    network : pandarm.Network
        pandarm Network object representing the study area
    distance : int
        Maximum distance (in units of geodataframe CRS) to consider the extent of the egohood
    decay : str
        type of decay function to apply. Options include
    precompute : bool
        Whether to precompute the pandarm Network object

    Attributes
    ----------
    statistic : float
        Rank-Order Information Theory Index
    core_data : a pandas DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.
    profile : pandas.DataFrame
        Segregation profile with one row per usable threshold, holding the
        threshold column name, the population share at or below it (`p`), and
        the pairwise index there (`statistic`)
    coefficients : numpy.ndarray
        Fitted polynomial coefficients, lowest order first
    deltas : numpy.ndarray
        Reardon's delta coefficients for the fitted polynomial degree
    degree : int
        Degree of the fitted polynomial
    standard_error : float
        Delta-method standard error of the statistic
    r_squared : float
        Weighted R-squared of the polynomial fit
    model : statsmodels results
        The fitted weighted least squares model, for further diagnostics

    Notes
    -----
    Reardon's :math:`H^R`, the information theory index integrated across the
    thresholds of an ordered variable using the entropy weight
    :math:`E(p) = -2[p \\ln p + (1 - p)\\ln(1 - p)]`, which is moderately
    sensitive to the extremes of the distribution.

    When a spatial argument is passed, every ordered category column is
    converted into an egohood count before the thresholds are computed, so the
    profile describes segregation between spatially-smoothed populations.

    Based on Reardon, Sean F. "Measures of income segregation."
    Working paper, Stanford Center for Education Policy Analysis (2011).

    Reference: :cite:`reardon2011measures`.

    See Also
    --------
    segregation.singlegroup.Entropy : the pairwise index integrated here.
    """

    def __init__(
        self,
        data,
        groups,
        degree=4,
        w=None,
        network=None,
        distance=None,
        decay="linear",
        function="triangular",
        precompute=False,
        **kwargs,
    ):
        """Init."""
        RankOrderIndex.__init__(self, data, groups)
        if any([w, network, distance]):
            SpatialImplicitIndex.__init__(
                self, w, network, distance, decay, function, precompute
            )
        aux = _rank_order_info_theory(self.data, self.groups, degree)

        self.statistic = aux[0]
        self.data = aux[1]
        self.groups = aux[2]
        self.profile = aux[3]
        self.deltas = aux[4]
        self.model = aux[5]
        self.coefficients = np.asarray(self.model.params)
        self.degree = int(degree)
        self.standard_error = float(
            np.sqrt(self.deltas @ self.model.cov_params() @ self.deltas)
        )
        self.r_squared = self.model.rsquared
        self._function = _rank_order_info_theory
