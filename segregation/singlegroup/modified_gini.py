"""Modified Gini Segregation Index."""

__author__ = "Renan X. Cortes <renanc@ucr.edu>, Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import geopandas as gpd
import numpy as np
import pandas as pd
from .._base import SingleGroupIndex, SpatialImplicitIndex
from .gini import _gini_seg
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import multiprocessing
from warnings import warn


def _modified_gini(
    data,
    group_pop_var,
    total_pop_var,
    iterations=500,
    backend="threading",
    n_jobs=-1,
    seed=None,
):
    """Calculate Modified Gini index.

    Parameters
    ----------
    data : pandas.DataFrame or geopandas.GeoDataFrame
        Dataframe or geodataframe if spatial index holding data for location of interest
    group_pop_var : string
        Variable containing the population count of the group of interest
    total_pop_var : string
        Variable in data that contains the total population count of the unit
    iterations : int
        The number of iterations the evaluate average classic dissimilarity under eveness.
        Default value is 500.
    n_jobs : int
        [Optional. Default=-1] Number of processes to run in parallel. If -1,
        this is set to the number of CPUs available
    backend : str {'loky', 'threading'}
        backend to pass into joblib's Parallel constructor.
    seed : int
        random seed passed to np.random inside the parallelization constructor to return
        consistent results


    Returns
    ----------
    statistic : float
        Modified Gini Index (Gini from Carrington and Troske (1997))
    data : pandas.DataFrame
        pandas DataFrame that contains the columns used to perform the estimate.

    Notes
    -----
    Based on Carrington, William J., and Kenneth R. Troske. "On measuring segregation in samples with small units." Journal of Business & Economic Statistics 15.4 (1997): 402-409.

    Reference: :cite:`carrington1997measuring`.
    """
    if not seed:
        seed = np.random.randint(
            0, 10e6
        )  # is there a better practice for this? I think joblib will fail if None passed
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    D = _gini_seg(data, group_pop_var, total_pop_var)[0]

    x = data[group_pop_var].values
    t = data[total_pop_var].values.astype(int)

    p_null = x.sum() / t.sum()

    # Ds = np.empty(iterations)

    def _gen_estimate(i):
        n_retries = 5
        try:
            while n_retries > 0:
                data = i[0]
                n = i[1]
                p = i[2]
                np.random.seed(i[3])
                freq_sim = np.random.binomial(
                    n=n,
                    p=p,
                    size=(1, data.shape[0]),
                ).tolist()[0]
                data[group_pop_var] = freq_sim
                aux = _gini_seg(data, group_pop_var, total_pop_var)[0]
                return aux

        except ValueError:
            warn("Simulator generated invalid data. Redrawing")
            n_retries -= 1

    Ds = pd.Series(
        Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_gen_estimate)(
                (
                    data,
                    np.array([t.tolist()]),
                    np.array([[p_null] * data.shape[0]]),
                    seed,
                )
            )
            for i in range(iterations)
        )
    )

    D_star = Ds.mean()

    if D >= D_star:
        Dct = (D - D_star) / (1 - D_star)
    else:
        Dct = (D - D_star) / D_star

    if not isinstance(data, gpd.GeoDataFrame):
        core_data = data[[group_pop_var, total_pop_var]]

    else:
        core_data = data[[group_pop_var, total_pop_var, data.geometry.name]]

    return Dct, core_data


class ModifiedGini(SingleGroupIndex, SpatialImplicitIndex):
    """Modified Gini Index.

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
        Modified Gini Index
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
        iterations=500,
        w=None,
        network=None,
        distance=None,
        decay="linear",
        function="triangular",
        precompute=None,
        backend="threading",
        n_jobs=-1,
        **kwargs
    ):
        """Init."""

        SingleGroupIndex.__init__(self, data, group_pop_var, total_pop_var)
        if any([w, network, distance]):
            SpatialImplicitIndex.__init__(
                self, w, network, distance, decay, function, precompute
            )
        aux = _modified_gini(
            self.data,
            self.group_pop_var,
            self.total_pop_var,
            iterations,
            backend=backend,
            n_jobs=n_jobs,
        )

        self.statistic = aux[0]
        self.data = aux[1]
        self._function = _modified_gini
