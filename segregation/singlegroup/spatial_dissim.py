"""Spatial Dissimilarity Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import libpysal
import numpy as np
from libpysal.weights import Queen

from .._base import SingleGroupIndex, SpatialExplicitIndex
from .dissim import _dissim


def _spatial_dissim(data, group_pop_var, total_pop_var, w=None, standardize=False):
    """Calculate of Spatial Dissimilarity index.

    Parameters
    ----------
    data : a geopandas DataFrame with a geometry column.
    group_pop_var : string
        The name of variable in data that contains the population size of the group of interest
    total_pop_var : string
        The name of variable in data that contains the total population of the unit
    w : W
        A PySAL weights object. If not provided, Queen contiguity matrix is used.
    standardize  : boolean
        A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
        For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
        works by default with row standardization.

    Returns
    ----------
    statistic : float
        Spatial Dissimilarity Index
    core_data : a geopandas DataFrame
        A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Morrill, R. L. (1991) "On the Measure of Geographic Segregation". Geography Research Forum.

    Reference: :cite:`morrill1991measure`.

    """
    if type(standardize) is not bool:
        raise TypeError("std is not a boolean object")

    if w is None:
        w_object = Queen.from_dataframe(data)
    else:
        w_object = w

    if not issubclass(type(w_object), libpysal.weights.W):
        raise TypeError("w is not a PySAL weights object")

    D = _dissim(data, group_pop_var, total_pop_var)[0]

    x = np.array(data[group_pop_var])
    t = np.array(data[total_pop_var])

    # If a unit has zero population, the group of interest frequency is zero
    pi = np.where(t == 0, 0, x / t)

    if not standardize:
        cij = w_object.sparse.toarray()
    else:
        cij = w_object.sparse.toarray()
        cij = cij / cij.sum(axis=1).reshape((cij.shape[0], 1))

    # Inspired in (second solution): https://stackoverflow.com/questions/22720864/efficiently-calculating-a-euclidean-distance-matrix-using-numpy
    # Distance Matrix
    abs_dist = abs(pi[..., np.newaxis] - pi)

    # manhattan_distances used to compute absolute distances
    num = np.multiply(abs_dist, cij).sum()
    den = cij.sum()
    SD = D - num / den
    SD

    core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return SD, core_data


class SpatialDissim(SingleGroupIndex, SpatialExplicitIndex):
    """Spatial Dissimilarity Index.

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
    standardize : boolean
        A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
        For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
        works by default with row standardization.

    Attributes
    ----------
    statistic : float
        SpatialDissim Index
    core_data : a pandas DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Morrill, R. L. (1991) "On the Measure of Geographic Segregation". Geography Research Forum.

    Reference: :cite:`morrill1991measure`.
    """

    def __init__(
        self, data, group_pop_var, total_pop_var, w=None, standardize=False, **kwargs
    ):
        """Init."""
        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        SpatialExplicitIndex.__init__(self,)
        self.w = w
        self.standardize = standardize
        aux = _spatial_dissim(
            self.data, self.group_pop_var, self.total_pop_var, self.w, self.standardize
        )

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _spatial_dissim
