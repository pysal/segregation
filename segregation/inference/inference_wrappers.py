"""
Inference Wrappers for Segregation measures
"""

__author__ = "Renan X. Cortes <renanc@ucr.edu> Sergio J. Rey <sergio.rey@ucr.edu> and Elijah Knaap <elijah.knaap@ucr.edu>"

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from segregation.util.util import _generate_counterfactual
from tqdm.auto import tqdm

from .._base import MultiGroupIndex, SingleGroupIndex
from .randomization import SIMULATORS, simulate_null

__all__ = [
    "SingleValueTest",
    "TwoValueTest",
]


def _infer_segregation(
    seg_class,
    iterations_under_null=500,
    null_approach="systematic",
    two_tailed=True,
    index_kwargs=None,
):
    """
    Perform inference for a single segregation measure

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
    index_kwargs : customizable parameters to pass to the segregation measures. Usually they need to be the same input that the seg_class was built.

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
    two_tailed : boolean. 
        If True, p_value is two-tailed. Otherwise, it is right one-tailed. The one-tailed p_value attribute 
        might not be appropriate for some measures, as the two-tailed. Therefore, it is better to rely on the
        est_sim attribute.
    **kwargs : dict
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

    def plot(self, ax=None):
        """
        Plot the Infer_Segregation class
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            warnings.warn("This method relies on importing `matplotlib` and `seaborn`")

        f = sns.kdeplot(
            self.est_sim,
            color="darkblue",
            linewidth=2,
            ax=ax,
        )
        plt.axvline(self.statistic, color="red")
        plt.title("{} (Value = {})".format(self._class_name, round(self.statistic, 3)))
        return f


def _compare_segregation(
    seg_class_1,
    seg_class_2,
    iterations_under_null=500,
    null_approach="random_label",
    **kwargs,
):
    """
    Perform inference comparison for a two segregation measures

    Parameters
    ----------

    seg_class_1           : a PySAL segregation object to be compared to seg_class_2
    
    seg_class_2           : a PySAL segregation object to be compared to seg_class_1
    
    iterations_under_null : number of iterations under null hyphothesis
    
    null_approach: argument that specifies which type of null hypothesis the inference will iterate.
    
        "random_label"               : random label the data in each iteration
        
        "counterfactual_composition" : randomizes the number of minority population according to both cumulative distribution function of a variable that represents the composition of the minority group. The composition is the division of the minority population of unit i divided by total population of tract i.

        "counterfactual_share" : randomizes the number of minority population and total population according to both cumulative distribution function of a variable that represents the share of the minority group. The share is the division of the minority population of unit i divided by total population of minority population.
        
        "counterfactual_dual_composition" : applies the "counterfactual_composition" for both minority and complementary groups.

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

    if not null_approach in [
        "random_label",
        "counterfactual_composition",
        "counterfactual_share",
        "counterfactual_dual_composition",
    ]:
        raise ValueError(
            "null_approach must one of 'random_label', 'counterfactual_composition', 'counterfactual_share', 'counterfactual_dual_composition'"
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

    est_sim = np.empty(iterations_under_null)

    ################
    # RANDOM LABEL #
    ################
    if null_approach == "random_label":

        data_1["grouping_variable"] = "Group_1"
        data_2["grouping_variable"] = "Group_2"

        if isinstance(seg_class_1, SingleGroupIndex):

            # This step is just to make sure the each frequecy column is integer for the approaches and from the same type in order to be able to stack them
            data_1.loc[:, (seg_class_1.group_pop_var, seg_class_1.total_pop_var)] = (
                data_1.loc[:, (seg_class_1.group_pop_var, seg_class_1.total_pop_var)]
                .round(0)
                .astype(int)
            )

            # random permutation needs the columns to have the same names
            data_1 = data_1[
                [
                    seg_class_1.group_pop_var,
                    seg_class_1.total_pop_var,
                    "grouping_variable",
                ]
            ]
            data_1.columns = ["group", "total", "grouping_variable"]

            data_2.loc[:, (seg_class_2.group_pop_var, seg_class_2.total_pop_var)] = (
                data_2.loc[:, (seg_class_2.group_pop_var, seg_class_2.total_pop_var)]
                .round(0)
                .astype(int)
            )
            data_2 = data_2[
                [
                    seg_class_2.group_pop_var,
                    seg_class_2.total_pop_var,
                    "grouping_variable",
                ]
            ]
            data_2.columns = ["group", "total", "grouping_variable"]

            stacked_data = pd.concat([data_1, data_2], axis=0)

            with tqdm(total=iterations_under_null) as pbar:
                for i in np.array(range(iterations_under_null)):

                    stacked_data["grouping_variable"] = np.random.permutation(
                        stacked_data["grouping_variable"].values
                    )

                    stacked_data_1 = stacked_data[
                        stacked_data["grouping_variable"] == "Group_1"
                    ]
                    stacked_data_2 = stacked_data[
                        stacked_data["grouping_variable"] == "Group_2"
                    ]

                    simulations_1 = seg_class_1._function(
                        stacked_data_1, "group", "total", **kwargs
                    )[0]
                    simulations_2 = seg_class_2._function(
                        stacked_data_2, "group", "total", **kwargs
                    )[0]

                    est_sim[i] = simulations_1 - simulations_2
                    pbar.set_description(
                        "Processed {} iterations out of {}".format(
                            i + 1, iterations_under_null
                        )
                    )
                    pbar.update(1)

        if isinstance(seg_class_1, MultiGroupIndex):

            groups_list = seg_class_1.groups

            for i in range(len(groups_list)):
                data_1[groups_list[i]] = round(data_1[groups_list[i]]).astype(int)
                data_2[groups_list[i]] = round(data_2[groups_list[i]]).astype(int)

            if seg_class_1.groups != seg_class_2.groups:
                raise ValueError("MultiGroup groups should be the same")

            stacked_data = pd.concat([data_1, data_2], ignore_index=True)

            with tqdm(total=iterations_under_null) as pbar:
                for i in np.array(range(iterations_under_null)):

                    stacked_data["grouping_variable"] = np.random.permutation(
                        stacked_data["grouping_variable"]
                    )

                    stacked_data_1 = stacked_data.loc[
                        stacked_data["grouping_variable"] == "Group_1"
                    ]
                    stacked_data_2 = stacked_data.loc[
                        stacked_data["grouping_variable"] == "Group_2"
                    ]

                    simulations_1 = seg_class_1._function(
                        stacked_data_1, groups_list, **kwargs
                    )[0]
                    simulations_2 = seg_class_2._function(
                        stacked_data_2, groups_list, **kwargs
                    )[0]

                    est_sim[i] = simulations_1 - simulations_2
                    pbar.set_description(
                        "Processed {} iterations out of {}".format(
                            i + 1, iterations_under_null
                        )
                    )
                    pbar.update(1)

    ##############################
    # COUNTERFACTUAL COMPOSITION #
    ##############################
    if null_approach in [
        "counterfactual_composition",
        "counterfactual_share",
        "counterfactual_dual_composition",
    ]:

        if isinstance(seg_class_1, MultiGroupIndex):
            raise ValueError("Not implemented for MultiGroup indexes.")

        internal_arg = null_approach[
            15:
        ]  # Remove 'counterfactual_' from the beginning of the string

        counterfac_df1, counterfac_df2 = _generate_counterfactual(
            data_1,
            data_2,
            seg_class_1.group_pop_var,
            seg_class_1.total_pop_var,
            seg_class_2.group_pop_var,
            seg_class_2.total_pop_var,
            counterfactual_approach=internal_arg,
        )

        if null_approach in ["counterfactual_share", "counterfactual_dual_composition"]:
            data_1[seg_class_1.total_pop_var] = counterfac_df1[
                "counterfactual_total_pop"
            ]
            data_2[seg_class_2.total_pop_var] = counterfac_df2[
                "counterfactual_total_pop"
            ]
        with tqdm(total=iterations_under_null) as pbar:
            for i in np.array(range(iterations_under_null)):

                data_1["fair_coin"] = np.random.uniform(size=len(data_1))
                data_1["test_group_pop_var"] = np.where(
                    data_1["fair_coin"] > 0.5,
                    data_1[seg_class_1.group_pop_var],
                    counterfac_df1["counterfactual_group_pop"],
                )

                # Dropping to avoid confusion in the internal function
                data_1_test = data_1.drop([seg_class_1.group_pop_var], axis=1)

                simulations_1 = seg_class_1._function(
                    data_1_test,
                    "test_group_pop_var",
                    seg_class_1.total_pop_var,
                    **kwargs,
                )[0]

                # Dropping to avoid confusion in the next iteration
                data_1 = data_1.drop(["fair_coin", "test_group_pop_var"], axis=1)

                data_2["fair_coin"] = np.random.uniform(size=len(data_2))
                data_2["test_group_pop_var"] = np.where(
                    data_2["fair_coin"] > 0.5,
                    data_2[seg_class_2.group_pop_var],
                    counterfac_df2["counterfactual_group_pop"],
                )

                # Dropping to avoid confusion in the internal function
                data_2_test = data_2.drop([seg_class_2.group_pop_var], axis=1)

                simulations_2 = seg_class_2._function(
                    data_2_test,
                    "test_group_pop_var",
                    seg_class_2.total_pop_var,
                    **kwargs,
                )[0]

                # Dropping to avoid confusion in the next iteration
                data_2 = data_2.drop(["fair_coin", "test_group_pop_var"], axis=1)

                est_sim[i] = simulations_1 - simulations_2

                pbar.set_description(
                    "Processed {} iterations out of {}".format(
                        i + 1, iterations_under_null
                    )
                )
                pbar.update(1)

    # Check and, if the case, remove iterations_under_null that resulted in nan or infinite values
    if any((np.isinf(est_sim) | np.isnan(est_sim))):
        warnings.warn(
            "Some estimates resulted in NaN or infinite values for estimations under null hypothesis. These values will be removed for the final results."
        )
        est_sim = est_sim[~(np.isinf(est_sim) | np.isnan(est_sim))]

    # Two-Tailed p-value
    # Obs.: the null distribution can be located far from zero. Therefore, this is the the appropriate way to calculate the two tailed p-value.
    aux1 = (point_estimation < est_sim).sum()
    aux2 = (point_estimation > est_sim).sum()
    p_value = 2 * np.array([aux1, aux2]).min() / len(est_sim)

    return p_value, est_sim, point_estimation, _class_name


class TwoValueTest:
    """
    Perform inference comparison for a two segregation measures

    Parameters
    ----------

    seg_class_1           : a PySAL segregation object to be compared to seg_class_2
    
    seg_class_2           : a PySAL segregation object to be compared to seg_class_1
    
    iterations_under_null : number of iterations under null hyphothesis
    
    null_approach : argument that specifies which type of null hypothesis the inference will iterate.
    
        "random_label"      : random label the data in each iteration
        
        "counterfactual_composition" : randomizes the number of minority population according to both cumulative distribution function of a variable that represents the composition of the minority group. The composition is the division of the minority population of unit i divided by total population of tract i.

        "counterfactual_share" : randomizes the number of minority population and total population according to both cumulative distribution function of a variable that represents the share of the minority group. The share is the division of the minority population of unit i divided by total population of minority population.
        
        "counterfactual_dual_composition" : applies the "counterfactual_composition" for both minority and complementary groups.

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
        **kwargs,
    ):

        aux = _compare_segregation(
            seg_class_1, seg_class_2, iterations_under_null, null_approach, **kwargs
        )

        self.p_value = aux[0]
        self.est_sim = aux[1]
        self.est_point_diff = aux[2]
        self._class_name = aux[3]

    def plot(self, ax=None):
        """
        Plot the Compare_Segregation class
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            warnings.warn("This method relies on importing `matplotlib` and `seaborn`")

        f = sns.distplot(
            self.est_sim,
            hist=True,
            color="darkblue",
            hist_kws={"edgecolor": "black"},
            kde_kws={"linewidth": 2},
            ax=ax,
        )
        plt.axvline(self.est_point_diff, color="red")
        plt.title(
            "{} (Diff. value = {})".format(
                self._class_name, round(self.est_point_diff, 3)
            )
        )
        return f
