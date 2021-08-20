"""Boundary Spatial Dissimilarity Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import numpy as np
from sklearn.metrics.pairwise import manhattan_distances

from .._base import (SingleGroupIndex, SpatialExplicitIndex,
                     _return_length_weighted_w)
from .dissim import _dissim


def _boundary_spatial_dissim(data, group_pop_var, total_pop_var, standardize=False):
    """Calculation of Boundary Spatial Dissimilarity index.

    Parameters
    ----------
    data : a geopandas DataFrame with a geometry column.
    group_pop_var : string
        The name of variable in data that contains the population size of the group of interest
    total_pop_var : string
        The name of variable in data that contains the total population of the unit
    standardize : boolean
        A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
        For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
        works by default without row standardization. That is, directly with border length.

    Returns
    ----------
    statistic : float
                Boundary Spatial Dissimilarity Index
    core_data : a geopandas DataFrame
                A geopandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    The formula is based on Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.

    Original paper by Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.

    References: :cite:`hong2014implementing` and :cite:`wong1993spatial`.

    """
    if type(standardize) is not bool:
        raise TypeError("std is not a boolean object")

    D = _dissim(data, group_pop_var, total_pop_var)[0]

    # If a unit has zero population, the group of interest frequency is zero
    data = data.assign(
        pi=np.where(
            data[total_pop_var] == 0, 0, data[group_pop_var] / data[total_pop_var]
        )
    )

    if not standardize:
        cij = _return_length_weighted_w(data).sparse.todense()
    else:
        cij = _return_length_weighted_w(data).sparse.todense()
        cij = cij / cij.sum(axis=1).reshape((cij.shape[0], 1))

    # manhattan_distances used to compute absolute distances
    num = np.multiply(manhattan_distances(data[["pi"]]), cij).sum()
    den = cij.sum()
    BSD = D - num / den

    core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return BSD, core_data


class BoundarySpatialDissim(SingleGroupIndex, SpatialExplicitIndex):
    """Boundary-Area Dissimilarity Index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    group_pop_var : str, required
        name of column on dataframe holding population totals for focal group
    total_pop_var : str, required
        name of column on dataframe holding total overall population
    standardize : boolean
        A condition for row standardisation of the weights matrices. If True, the values of cij in the formulas gets row standardized.
        For the sake of comparison, the seg R package of Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.
        works by default with row standardization.

    Attributes
    ----------
    statistic : float
        Boundary Area Index
    core_data : a pandas DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    The formula is based on Hong, Seong-Yun, David O'Sullivan, and Yukio Sadahiro. "Implementing spatial segregation measures in R." PloS one 9.11 (2014): e113767.

    Original paper by Wong, David WS. "Spatial indices of segregation." Urban studies 30.3 (1993): 559-572.

    References: :cite:`hong2014implementing` and :cite:`wong1993spatial`.
    """

    def __init__(
        self, data, group_pop_var, total_pop_var, w=None, standardize=True, **kwargs
    ):
        """Init."""
        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        SpatialExplicitIndex.__init__(self,)
        self.standardize = standardize
        aux = _boundary_spatial_dissim(
            self.data, self.group_pop_var, self.total_pop_var, self.standardize
        )

        self.statistic = aux[0]
        self.core_data = aux[1]
        self._function = _boundary_spatial_dissim
