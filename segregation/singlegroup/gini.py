"""Gini Segregation Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import geopandas as gpd
import numpy as np

from .._base import SingleGroupIndex, SpatialImplicitIndex



try:
    from numba import njit, jit, prange, boolean
except (ImportError, ModuleNotFoundError):

    def jit(*dec_args, **dec_kwargs):
        """
        decorator mimicking numba.jit
        """

        def intercepted_function(f, *f_args, **f_kwargs):
            return f

        return intercepted_function

    njit = jit

    prange = range
    boolean = bool

@njit(parallel=True, fastmath=True,)
def _gini_vecp(pi: np.ndarray, ti: np.ndarray):
    """Memory efficient calculation of Gini

    Parameters
    ----------
    pi : np.ndarray
        area minority population counts
    ti : np.ndarray
        area total population counts

    Returns
    ----------
    
    implicit: float
             Gini coefficient
    """


    n = ti.shape[0]
    num = np.zeros(1)
    T = ti.sum()
    P = pi.sum() / T
    pi = np.where(ti == 0, 0, pi / ti)
    T = ti.sum()
    for i in prange(n-1):
        num += (ti[i] * ti[i+1:] * np.abs(pi[i] - pi[i+1:])).sum()
    num *= 2
    den = (2 * T * T * P * (1-P))
    return (num / den)[0]

    

def _gini_seg(data, group_pop_var, total_pop_var):
    """Calculate Gini segregation index.

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
        MinMax index statistic value
    core_data : pandas.DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    Reference: :cite:`massey1988dimensions`.
    """

    # If a unit has zero population, the group of interest frequency is zero
    data = data.assign(
        ti=data[total_pop_var],
        pi=np.where(
            data[total_pop_var] == 0, 0, data[group_pop_var] / data[total_pop_var]
        ),
    )

    pi = data[group_pop_var].values
    ti = data[total_pop_var].values
    G = _gini_vecp(pi, ti)

    if not isinstance(data, gpd.GeoDataFrame):
        data = data[[group_pop_var, total_pop_var]]

    else:
        data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return G, data

class Gini(SingleGroupIndex, SpatialImplicitIndex):
    """Gini Index.

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
                Gini Index
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Massey, Douglas S., and Nancy A. Denton. "The dimensions of residential segregation." Social forces 67.2 (1988): 281-315.

    Reference: :cite:`massey1988dimensions`.
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
        **kwargs
    ):
        """Init."""
        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        if any([w, network, distance]):
            SpatialImplicitIndex.__init__(
                self, w, network, distance, decay, function, precompute
            )
        aux = _gini_seg(self.data, self.group_pop_var, self.total_pop_var)

        self.statistic = aux[0]
        self.data = aux[1]
        self._function = _gini_seg
