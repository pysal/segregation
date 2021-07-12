"""Inference wrapper classes for segregation measures."""

__author__ = "Renan X. Cortes <renanc@ucr.edu> Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import multiprocessing
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .._base import MultiGroupIndex
from .comparative import (DUAL_SIMULATORS, _estimate_counterfac_difference,
                          _estimate_random_label_difference,
                          _generate_counterfactual, _prepare_random_label)
from .randomization import SIMULATORS, simulate_null


def _infer_segregation(
    seg_class,
    iterations_under_null=500,
    null_approach="systematic",
    two_tailed=True,
    index_kwargs=None,
    n_jobs=-1,
    backend="loky",
):
    """Perform inference for a single segregation measure.

    Parameters
    ----------
    seg_class : a PySAL segregation object
        fitted segregation class
    iterations_under_null : int
        number of iterations under null hyphothesis
    null_approach : argument that specifies which type of null hypothesis the inference will iterate. Please take a look at Notes (1).
        "systematic" : assumes that every group has the same probability with restricted conditional probabilities p_0_j = p_1_j = p_j = n_j/n (multinomial distribution).
        "bootstrap" : generates bootstrap replications of the units with replacement of the same size of the original data.
        "evenness" : assumes that each spatial unit has the same global probability of drawing elements from the minority group of the fixed total unit population (binomial distribution).
        "permutation" : randomly allocates the units over space keeping the original values.
        "systematic_permutation" : assumes absence of systematic segregation and randomly allocates the units over space.
        "even_permutation" : assumes the same global probability of drawning elements from the minority group in each spatial unit and randomly allocates the units over space.
    two_tailed : boolean. Please take a look at Notes (2).
        If True, p_value is two-tailed. Otherwise, it is right one-tailed.
    n_jobs: int, optional
        number of cores to use for estimation. If -1 all available cpus will be used
    backend: str, optional
        which backend to use with joblib. Options include "loky", "multiprocessing", or "threading"
    index_kwargs : dict, optional
        customizable parameters to pass to the segregation measures. Usually they need to be the same input that the seg_class was built.

    Attributes
    ----------
    p_value     : float
        Pseudo One or Two-Tailed p-value estimated from the simulations
    est_sim     : numpy array
        Estimates of the segregation measure under the null hypothesis
    statistic   : float
        The point estimation of the segregation measure that is under test

    Notes
    -----
    1) The different approaches for the null hypothesis affect directly the results of the inference depending on the combination of the index type of seg_class and the null_approach chosen.
    Therefore, the user needs to be aware of how these approaches are affecting the data generation process of the simulations in order to draw meaningful conclusions. 
    For example, the Modified Dissimilarity (ModifiedDissim) and  Modified Gini (ModifiedGiniSeg) indexes, rely exactly on the distance between evenness through sampling which, therefore, the "evenness" value for null approach would not be the most appropriate for these indexes.

    2) The one-tailed p_value attribute might not be appropriate for some measures, as the two-tailed. Therefore, it is better to rely on the est_sim attribute.

    """
    if null_approach not in SIMULATORS.keys():
        raise ValueError(f"null_approach must one of {list(SIMULATORS.keys())}")

    if type(two_tailed) is not bool:
        raise TypeError("two_tailed is not a boolean object")

    point_estimation = seg_class.statistic

    aux = str(type(seg_class))
    _class_name = aux[
        1 + aux.rfind(".") : -2
    ]  # 'rfind' finds the last occurence of a pattern in a string

    Estimates_Stars = simulate_null(
        iterations=iterations_under_null,
        sim_func=SIMULATORS[null_approach],
        seg_class=seg_class,
        index_kwargs=index_kwargs,
        n_jobs=n_jobs,
        backend=backend,
    ).values

    # Check and, if the case, remove iterations_under_null that resulted in nan or infinite values
    if any((np.isinf(Estimates_Stars) | np.isnan(Estimates_Stars))):
        warnings.warn(
            "Some estimates resulted in NaN or infinite values for estimations under null hypothesis. These values will be removed for the final results."
        )
        Estimates_Stars = Estimates_Stars[
            ~(np.isinf(Estimates_Stars) | np.isnan(Estimates_Stars))
        ]

    if not two_tailed:
        p_value = sum(Estimates_Stars > point_estimation) / iterations_under_null
    else:
        aux1 = (point_estimation < Estimates_Stars).sum()
        aux2 = (point_estimation > Estimates_Stars).sum()
        p_value = 2 * np.array([aux1, aux2]).min() / len(Estimates_Stars)

    return p_value, Estimates_Stars, point_estimation, _class_name


class SingleValueTest:
    """Statistical inference for a single segregation measure.

    Parameters
    ----------
    seg_class : segregation.singlegroup or segregation.multigroup object
        fitted segregation index class
    iterations_under_null : int
        number of iterations under null hyphothesis
    null_approach : str
        Which counterfactual approach to use when generating null hypothesis distribution. Please take a look at Notes (1).

            * "systematic" : assumes that every group has the same probability with restricted conditional probabilities p_0_j = p_1_j = p_j = n_j/n (multinomial distribution).
            * "bootstrap" : generates bootstrap replications of the units with replacement of the same size of the original data.
            * "evenness" : assumes that each spatial unit has the same global probability of drawing elements from the minority group of the fixed total unit population (binomial distribution).
            * "person_permutation" : randomly allocates individuals into units keeping the total population of each equal to the original.
            * "geographic_permutation" : randomly allocates the units over space keeping the original values.
            * "systematic_permutation" : assumes absence of systematic segregation and randomly allocates the units over space.
            * "even_permutation" : assumes the same global probability of drawning elements from the minority group in each spatial unit and randomly allocates the units over space.
    two_tailed : boolean
        If True, p_value is two-tailed. Otherwise, it is right one-tailed. The one-tailed p_value attribute 
        might not be appropriate for some measures, as the two-tailed. Therefore, it is better to rely on the
        est_sim attribute.
    n_jobs: int, optional
        number of cores to use for estimation. If -1 all available cpus will be used
    backend: str, optional
        which backend to use with joblib. Options include "loky", "multiprocessing", or "threading"
    index_kwargs : dict, optional
        customizable parameters to pass to the segregation measures. Usually they need to be the same input that the seg_class was built.

    Attributes
    ----------
    p_value : float
        Pseudo One or Two-Tailed p-value estimated from the simulations
    est_sim : numpy array
       Estimates of the segregation measure under the null hypothesis
    statistic : float
        The point estimate of the segregation measure that is under test

    Notes
    -----
    1) The different approaches for the null hypothesis affect directly the results of the inference depending on the combination of the index type of seg_class and the null_approach chosen.
    Therefore, the user needs to be aware of how these approaches are affecting the data generation process of the simulations in order to draw meaningful conclusions. 
    For example, the Modified Dissimilarity (ModifiedDissim) and  Modified Gini (ModifiedGiniSeg) indexes, rely exactly on the distance between evenness through sampling which, therefore, the "evenness" value for null approach would not be the most appropriate for these indexes.

    Examples
    --------
    Several examples can be found here https://github.com/pysal/segregation/blob/master/notebooks/inference_wrappers_example.ipynb.
    """

    def __init__(
        self,
        seg_class,
        iterations_under_null=500,
        null_approach="systematic",
        two_tailed=True,
        **kwargs,
    ):

        aux = _infer_segregation(
            seg_class, iterations_under_null, null_approach, two_tailed, **kwargs
        )

        self.p_value = aux[0]
        self.est_sim = aux[1]
        self.statistic = aux[2]
        self._class_name = aux[3]

    def plot(self, color="darkblue", kde=True, ax=None, **kwargs):
        """Plot the distribution of simulated values and the index value being tested.

        Parameters
        ----------
        color : str, optional
            color of histogram, by default 'darkblue'
        kde : bool, optional
            Whether to plot the kernel density estimate along with the histogram, by default True
        ax : matplotlib.axes, optional
            axes object to plot onto, by default None
        kwargs : seaborn.histplot argument, optional
            additional keyword arguments passed to seaborn's histplot function

        Returns
        -------
        matplotlib.axes
            pyplot axes object
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            warnings.warn("This method relies on importing `matplotlib` and `seaborn`")

        f = sns.histplot(self.est_sim, color=color, kde=kde, ax=ax, **kwargs)
        plt.axvline(self.statistic, color="red")
        plt.title("{} (Value = {})".format(self._class_name, round(self.statistic, 3)))
        return f


def _compare_segregation(
    seg_class_1,
    seg_class_2,
    iterations=500,
    null_approach="random_label",
    index_kwargs_1=None,
    index_kwargs_2=None,
    n_jobs=-1,
    backend="loky",
):
    """Perform inference comparison for a two segregation measures

    Parameters
    ----------

    seg_class_1           : a PySAL segregation object to be compared to seg_class_2
    
    seg_class_2           : a PySAL segregation object to be compared to seg_class_1
    
    iterations_under_null : number of iterations under null hyphothesis
    
    null_approach: argument that specifies which type of null hypothesis the inference will iterate.
    
        "random_label"               : random label the data in each iteration
        
        "composition" : randomizes the number of minority population according to both cumulative distribution function of a variable that represents the composition of the minority group. The composition is the division of the minority population of unit i divided by total population of tract i.

        "share" : randomizes the number of minority population and total population according to both cumulative distribution function of a variable that represents the share of the minority group. The share is the division of the minority population of unit i divided by total population of minority population.
        
        "composition" : applies the "counterfactual_composition" for both minority and complementary groups.

    **kwargs : customizable parameters to pass to the segregation measures. Usually they need to be the same as both seg_class_1 and seg_class_2  was built.
    
    Attributes
    ----------

    p_value        : float
                     Two-Tailed p-value
    
    est_sim        : numpy array
                     Estimates of the segregation measure differences under the null hypothesis
                  
    est_point_diff : float
                     Point estimation of the difference between the segregation measures
                
    Notes
    -----
    This function performs inference to compare two segregation measures. This can be either two measures of the same locations in two different points in time or it can be two different locations at the same point in time.
    
    The null hypothesis is H0: Segregation_1 is not different than Segregation_2.
    
    Based on Rey, Sergio J., and Myrna L. Sastré-Gutiérrez. "Interregional inequality dynamics in Mexico." Spatial Economic Analysis 5.3 (2010): 277-298.
    """

    if not index_kwargs_1:
        index_kwargs_1 = {}
    if not index_kwargs_2:
        index_kwargs_2 = {}
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    if null_approach not in [
        "random_label",
        "composition",
        "share",
        "dual_composition",
    ]:
        raise ValueError(
            f"null_approach must one of {list(DUAL_SIMULATORS.keys())+['random_label']}"
        )

    if type(seg_class_1) != type(seg_class_2):
        raise TypeError("seg_class_1 and seg_class_2 must be the same type/class.")

    point_estimation = seg_class_1.statistic - seg_class_2.statistic

    aux = str(type(seg_class_1))
    _class_name = aux[
        1 + aux.rfind(".") : -2
    ]  # 'rfind' finds the last occurence of a pattern in a string

    data_1 = seg_class_1.data.copy()
    data_2 = seg_class_2.data.copy()

    if null_approach == "random_label":
        if isinstance(seg_class_1, MultiGroupIndex):
            groups = seg_class_1.groups
        else:
            groups = None

        stacked = _prepare_random_label(seg_class_1, seg_class_2)

        estimates = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_estimate_random_label_difference)(
                (
                    stacked,
                    seg_class_1._function,
                    index_kwargs_1,
                    index_kwargs_2,
                    seg_class_1.index_type,
                    groups,
                )
            )
            for i in tqdm(range(iterations))
        )

    if null_approach in [
        "composition",
        "share",
        "dual_composition",
    ]:

        if isinstance(seg_class_1, MultiGroupIndex):
            raise ValueError("Not implemented for MultiGroup indexes.")

        counterfac_df1, counterfac_df2 = _generate_counterfactual(
            data_1,
            data_2,
            seg_class_1.group_pop_var,
            seg_class_1.total_pop_var,
            seg_class_2.group_pop_var,
            seg_class_2.total_pop_var,
            null_approach,
        )

        if null_approach in ["share", "dual_composition"]:
            data_1[seg_class_1.total_pop_var] = counterfac_df1[
                "counterfactual_total_pop"
            ]
            data_2[seg_class_2.total_pop_var] = counterfac_df2[
                "counterfactual_total_pop"
            ]

        estimates = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_estimate_counterfac_difference)(
                (
                    data_1,
                    data_2,
                    seg_class_1.group_pop_var,
                    seg_class_1.total_pop_var,
                    seg_class_2.group_pop_var,
                    seg_class_2.total_pop_var,
                    index_kwargs_1,
                    index_kwargs_2,
                    null_approach,
                    seg_class_1._function,
                    counterfac_df1,
                    counterfac_df2,
                )
            )
            for i in tqdm(range(iterations))
        )
    estimates = pd.Series(estimates).dropna()
    if len(estimates) < iterations:
        warnings.warn("Some observations were removed for NA values")

    # Two-Tailed p-value
    # Obs.: the null distribution can be located far from zero. Therefore, this is the the appropriate way to calculate the two tailed p-value.
    aux1 = (point_estimation < estimates).sum()
    aux2 = (point_estimation > estimates).sum()
    p_value = 2 * np.array([aux1, aux2]).min() / len(estimates)

    return p_value, estimates, point_estimation, _class_name


class TwoValueTest:
    """Perform comparative inferencefor two segregation measures.

    Parameters
    ----------
    seg_class_1 : segregation.singlegroup or segregation.multigroup class
        a fitted segregation class to be compared to seg_class_2
    seg_class_2 :
        a fitted segregation class to be compared to seg_class_1
    iterations_under_null : int
        number of iterations under null hyphothesis
    null_approach : str
        which type of null hypothesis the inference will iterate.

            * "random_label" : random label the data in each iteration
            * "composition" : randomizes the number of minority population according to both cumulative distribution function of a variable that represents the composition of the minority group. The composition is the division of the minority population of unit i divided by total population of tract i.
            * "share" : randomizes the number of minority population and total population according to both cumulative distribution function of a variable that represents the share of the minority group. The share is the division of the minority population of unit i divided by total population of minority population.
            * "dual_composition" : applies the "counterfactual_composition" for both minority and complementary groups.
    n_jobs: int, optional
        number of cores to use for estimation. If -1 all available cpus will be used
    backend: str, optional
        which backend to use with joblib. Options include "loky", "multiprocessing", or "threading"
    index_kwargs_1 : dict, optional
        extra parameters to pass to segregation index 1.
    index_kwargs_2 : dict, optional
        extra parameters to pass to segregation index 2.

    Attributes
    ----------
    p_value : float
        Two-Tailed p-value
    est_sim : numpy array
        Estimates of the segregation measure differences under the null hypothesis
    est_point_diff : float
        Point estimation of the difference between the segregation measures

    Notes
    -----
    This function performs inference to compare two segregation measures. This can be either
    two measures of the same locations in two different points in time or it can be two
    different locations at the same point in time. The null hypothesis is H0: Segregation_1
    is not different than Segregation_2.
    Based on Rey, Sergio J., and Myrna L. Sastré-Gutiérrez. "Interregional inequality dynamics in Mexico." Spatial Economic Analysis 5.3 (2010): 277-298.

    Examples
    --------
    Several examples can be found here https://github.com/pysal/segregation/blob/master/notebooks/inference_wrappers_example.ipynb.
    """

    def __init__(
        self,
        seg_class_1,
        seg_class_2,
        iterations_under_null=500,
        null_approach="random_label",
        n_jobs=-1,
        backend="loky",
        index_kwargs_1=None,
        index_kwargs_2=None,
        **kwargs,
    ):

        aux = _compare_segregation(
            seg_class_1,
            seg_class_2,
            iterations_under_null,
            null_approach,
            n_jobs=n_jobs,
            backend=backend,
            index_kwargs_1=index_kwargs_1,
            index_kwargs_2=index_kwargs_2,
        )

        self.p_value = aux[0]
        self.est_sim = aux[1]
        self.est_point_diff = aux[2]
        self._class_name = aux[3]

    def plot(self, color="darkblue", kde=True, ax=None, **kwargs):
        """Plot the distribution of simulated values and the index value being tested.

        Parameters
        ----------
        color : str, optional
            histogram color, by default 'darkblue'
        kde : bool, optional
            Whether to plot the kernel density estimate along with the histogram,
            by default True
        ax : matplotlib.axes, optional
            axes object to plot onto, by default None
        kwargs : seaborn.histplot argument, optional
            additional keyword arguments passed to seaborn's histplot function

        Returns
        -------
        matplotlib.axes
            pyplot axes object
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            warnings.warn("This method relies on importing `matplotlib` and `seaborn`")

        f = sns.histplot(self.est_sim, color=color, kde=kde, ax=ax, **kwargs)
        plt.axvline(self.est_point_diff, color="red")
        plt.title(
            "{} (Diff. value = {})".format(
                self._class_name, round(self.est_point_diff, 3)
            )
        )
        return f
