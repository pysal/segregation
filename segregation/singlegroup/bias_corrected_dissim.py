"""Bias Corrected Dissimilarity index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import geopandas as gpd
import numpy as np
import pandas as pd

from .._base import SingleGroupIndex, SpatialImplicitIndex
from .dissim import _dissim


def _bias_corrected_dissim(data, group_pop_var, total_pop_var, B=500):
    """
    Calculation of Bias Corrected Dissimilarity index

    Parameters
    ----------

    data : pandas.DataFrame or geopandas.GeoDataFrame
        DataFrame holding necessary data
    group_pop_var : string
        The name of variable in data that contains the population size of the group of interest
    total_pop_var : string
        The name of variable in data that contains the total population of the unit
    B : int
       The number of iterations to calculate Dissimilarity simulating randomness with multinomial distributions. Default value is 500.

    Returns
    ----------
    statistic : float
        Dissimilarity with Bias-Correction (bias correction from Allen, Rebecca et al. (2015))
    core_data : a pandas DataFrame
        A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Allen, Rebecca, et al. "More reliable inference for the dissimilarity index of segregation." The econometrics journal 18.1 (2015): 40-66.

    Reference: :cite:`allen2015more`.
    """
    assert type(B) is int, "B must be an integer"

    assert B > 1, "B must be greater than 1."

    D = _dissim(data, group_pop_var, total_pop_var)[0]

    x = np.array(data[group_pop_var])
    t = np.array(data[total_pop_var])

    other_group_pop = t - x

    # Group 0: minority group
    p0_i = x / x.sum()
    n0 = x.sum()
    sim0 = np.random.multinomial(n0, p0_i, size=B)

    # Group 1: complement group
    p1_i = other_group_pop / other_group_pop.sum()
    n1 = other_group_pop.sum()
    sim1 = np.random.multinomial(n1, p1_i, size=B)

    Dbcs = np.empty(B)
    for i in np.array(range(B)):
        data_aux = {
            "simul_group": sim0[i].tolist(),
            "simul_tot": (sim0[i] + sim1[i]).tolist(),
        }
        df_aux = pd.DataFrame.from_dict(data_aux)
        Dbcs[i] = _dissim(df_aux, "simul_group", "simul_tot")[0]

    Db = Dbcs.mean()

    Dbc = 2 * D - Db
    Dbc  # It expected to be lower than D, because D is upwarded biased

    if isinstance(data, gpd.GeoDataFrame):
        core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    else:
        core_data = data[[group_pop_var, total_pop_var]]

    return Dbc, core_data


class BiasCorrectedDissim(SingleGroupIndex, SpatialImplicitIndex):
    """Bias Corrected Dissimilarity Index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame, required
        dataframe or geodataframe if spatial index holding data for location of interest
    group_pop_var : str, required
        name of column on dataframe holding population totals for focal group
    total_pop_var : str, required
        name of column on dataframe holding total overall population
    B : int
       The number of iterations to calculate Dissimilarity simulating randomness with multinomial distributions. Default value is 500.
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
                BiasCorrectedDissim Index
    core_data : a pandas DataFrame
                A pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Carrington, William J., and Kenneth R. Troske. "On measuring segregation in samples with small units." Journal of Business & Economic Statistics 15.4 (1997): 402-409.

    Reference: :cite:`carrington1997measuring`.
    """

    def __init__(
        self,
        data,
        group_pop_var,
        total_pop_var,
        B=500,
        w=None,
        network=None,
        distance=None,
        decay=None,
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
        self.B = B
        aux = _bias_corrected_dissim(
            self.data, self.group_pop_var, self.total_pop_var, self.B
        )

        self.statistic = aux[0]
        self.data = aux[1]
        self._function = _bias_corrected_dissim
