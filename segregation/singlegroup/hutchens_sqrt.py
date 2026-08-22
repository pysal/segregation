"""Hutchens Square Root Segregation Index."""


import geopandas as gpd
import numpy as np

np.seterr(divide="ignore", invalid="ignore")

from .._base import SingleGroupIndex, SpatialImplicitIndex


def _hutchens_sqrt(data, group_pop_var, total_pop_var):
    """Calculate the Hutchens square root segregation index.

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
        Hutchens square root index statistic value
    core_data : pandas.DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    The index is defined as

    .. math::
        O_s = 1 - \\sum_j \\sqrt{\\frac{x_{1j}}{N_1}} \\sqrt{\\frac{x_{2j}}{N_2}}

    where :math:`x_{1j}` is the count of the focal group in unit :math:`j`,
    :math:`x_{2j}` the count of the complementary group, and :math:`N_1, N_2`
    the corresponding totals. It takes values in [0, 1], where 0 is complete
    integration and 1 is complete segregation.

    Unlike Dissim, this index satisfies the neighborhood division property (a
    Pigou-Dalton style transfer principle), placing it in the same axiomatic
    class as Gini. It is also symmetric in types: swapping the labels of the
    two groups leaves the statistic unchanged.

    Rewriting in terms of unit shares :math:`p_j = x_{1j}/t_j` and the global
    share :math:`P` shows this is identical to the "square root index" that
    Reardon integrates to form the rank-order index
    :class:`segregation.rankorder.RankOrderSqrt`:

    .. math::
        O_s = 1 - \\sum_j \\frac{t_j}{T}
              \\frac{\\sqrt{p_j (1 - p_j)}}{\\sqrt{P (1 - P)}}

    Based on Hutchens, Robert M. "One measure of segregation."
    International Economic Review 45.2 (2004): 555-577.

    Reference: :cite:`hutchens2004one`.
    """
    x1 = np.asarray(data[group_pop_var], dtype=float)
    x2 = np.asarray(data["group_2_pop_var"], dtype=float)
    t = np.asarray(data[total_pop_var], dtype=float)

    # `_function` is called directly by the decomposition and inference
    # machinery, which bypasses SingleGroupIndex validation, so re-check here
    if any(t < x1):
        raise ValueError(
            "Group of interest population must equal or lower than the total population of the units."
        )

    N1 = x1.sum()
    N2 = x2.sum()

    if (N1 == 0) or (N2 == 0):
        # no focal group, or the focal group is the entire population:
        # there is no segregation to measure
        O_s = 0.0
    else:
        # units with zero population contribute sqrt(0) * sqrt(0) = 0
        O_s = 1.0 - np.sum(np.sqrt(x1 / N1) * np.sqrt(x2 / N2))

    if not isinstance(data, gpd.GeoDataFrame):
        core_data = data[[group_pop_var, total_pop_var]]
    else:
        core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return O_s, core_data


class HutchensSqrt(SingleGroupIndex, SpatialImplicitIndex):
    """Hutchens Square Root Index.

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
                Hutchens Square Root Index
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Hutchens, Robert M. "One measure of segregation."
    International Economic Review 45.2 (2004): 555-577.

    Reference: :cite:`hutchens2004one`.

    See Also
    --------
    segregation.rankorder.RankOrderSqrt : integrates this index across the
        thresholds of an ordered variable to form Reardon's :math:`S^R`.
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
        **kwargs,
    ):
        """Init."""
        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        if any([w, network, distance]):
            SpatialImplicitIndex.__init__(
                self, w, network, distance, decay, function, precompute
            )
        aux = _hutchens_sqrt(self.data, self.group_pop_var, self.total_pop_var)

        self.statistic = aux[0]
        self.data = aux[1]
        self._function = _hutchens_sqrt
