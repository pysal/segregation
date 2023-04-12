"""Tools for simulating comparative datasets across spatial contexts."""

import numpy as np
import pandas as pd

from .._base import MultiGroupIndex, SingleGroupIndex
from .randomization import simulate_person_permutation


def _prepare_comparative_data(df1, df2, group_pop_var1, group_pop_var2, total_pop_var1, total_pop_var2):
    df1 = df1.copy()
    df2 = df2.copy()
    if hasattr(df1, "geometry"):
        df1 = df1[[group_pop_var1, total_pop_var1, df1.geometry.name]]
    else:
        df1 = df1[[group_pop_var1, total_pop_var1]]

    if hasattr(df2, "geometry"):
        df2 = df2[[group_pop_var2, total_pop_var2, df2.geometry.name]]
    else:
        df2 = df2[[group_pop_var2, total_pop_var2]]

    return df1, df2



def _generate_counterfactual(
    data1,
    data2,
    group_pop_var1,
    total_pop_var1,
    group_pop_var2,
    total_pop_var2,
    counterfactual_approach="composition",
):
    """Generate a counterfactual variables.

    Given two contexts, generate counterfactual distributions for a variable of
    interest by simulating the variable of one context into the spatial
    structure of the other.

    Parameters
    ----------
    data1 : pd.DataFrame or gpd.DataFrame
        Pandas or Geopandas dataframe holding data for context 1

    data2 : pd.DataFrame or gpd.DataFrame
        Pandas or Geopandas dataframe holding data for context 2

    group_pop_var : str
        The name of variable in both data that contains the population size of the group of interest

    total_pop_var : str
        The name of variable in both data that contains the total population of the unit

    approach : str, ["composition", "share", "dual_composition"]
        Which approach to use for generating the counterfactual.
        Options include "composition", "share", or "dual_composition"

    Returns
    -------
    two DataFrames
        df1 and df2  with appended columns 'counterfactual_group_pop', 'counterfactual_total_pop', 'group_composition' and 'counterfactual_composition'

    """
    df1, df2 = DUAL_SIMULATORS[counterfactual_approach](
        data1, data2, group_pop_var1, total_pop_var1, group_pop_var2, total_pop_var2,
    )
    df1["group_composition"] = (df1[group_pop_var1] / df1[total_pop_var1]).fillna(0)
    df2["group_composition"] = (df2[group_pop_var2] / df2[total_pop_var2]).fillna(0)

    df1["counterfactual_composition"] = (
        df1["counterfactual_group_pop"] / df1["counterfactual_total_pop"]
    ).fillna(0)
    df2["counterfactual_composition"] = (
        df2["counterfactual_group_pop"] / df2["counterfactual_total_pop"]
    ).fillna(0)

    df1 = df1.drop(columns=[group_pop_var1, total_pop_var1], axis=1)
    df2 = df2.drop(columns=[group_pop_var2, total_pop_var2], axis=1)

    return df1, df2


def sim_composition(
    df1, df2, group_pop_var1, total_pop_var1, group_pop_var2, total_pop_var2,
):
    """Simulate the spatial distribution of a population group in a region using the CDF of a comparison region.

    For each spatial unit i in region 1, take the unit's percentile in the distribution, and swap the group composition
    with the value of the corresponding percentile in region 2. The composition is the minority population of unit i
    divided by total population of tract i. This approach will shift the relative composition of each spatial
    unit without changing its total population.

    Parameters
    ----------
    df1 : pandas.DataFrame or geopandas.GeoDataFrame
        dataframe for first dataset with columns holding group and total population counts
    df2 : pandas.DataFrame or geopandas.GeoDataFrame
        dataframe for second dataset with columns holding group and total population counts
    group_pop_var1 : str
        column holding population counts for group of interest on input df1
    total_pop_var1 : str
        column holding total population counts on input df1
    group_pop_var2 : str
        column holding population counts for group of interest on input df2
    total_pop_var2 : str
        column holding total population counts on input df2

    Returns
    -------
    two pandas.DataFrame
        dataframes with simulated population columns appended
    """
    df1, df2 = _prepare_comparative_data(df1, df2, group_pop_var1, group_pop_var2, total_pop_var1, total_pop_var2)

    df1["group_composition"] = (df1[group_pop_var1] / df1[total_pop_var1]).fillna(0)
    df2["group_composition"] = (df2[group_pop_var2] / df2[total_pop_var2]).fillna(0)

    df1["counterfactual_group_pop"] = (
        df1["group_composition"].rank(pct=True).apply(df2["group_composition"].quantile)
        * df1[total_pop_var1]
    )
    df2["counterfactual_group_pop"] = (
        df2["group_composition"].rank(pct=True).apply(df1["group_composition"].quantile)
        * df2[total_pop_var2]
    )

    df1["counterfactual_total_pop"] = df1[total_pop_var1]
    df2["counterfactual_total_pop"] = df2[total_pop_var2]

    return df1, df2


def sim_dual_composition(
    df1, df2, group_pop_var1, total_pop_var1, group_pop_var2, total_pop_var2,
):
    """Apply the 'composition' for both minority and complementary groups.

    Parameters
    ----------
    df1 : pandas.DataFrame or geopandas.GeoDataFrame
        dataframe for first dataset with columns holding group and total population counts
    df2 : pandas.DataFrame or geopandas.GeoDataFrame
        dataframe for second dataset with columns holding group and total population counts
    group_pop_var1 : str
        column holding population counts for group of interest on input df1
    total_pop_var1 : str
        column holding total population counts on input df1
    group_pop_var2 : str
        column holding population counts for group of interest on input df2
    total_pop_var2 : str
        column holding total population counts on input df2

    Returns
    -------
    two pandas.DataFrame
        dataframes with simulated population columns appended

    """
    df1, df2 = _prepare_comparative_data(df1, df2, group_pop_var1, group_pop_var2, total_pop_var1, total_pop_var2)

    df1["group_composition"] = (df1[group_pop_var1] / df1[total_pop_var1]).fillna(0)
    df2["group_composition"] = (df2[group_pop_var2] / df2[total_pop_var2]).fillna(0)

    df1["compl_pop_var"] = df1[total_pop_var1] - df1[group_pop_var1]
    df2["compl_pop_var"] = df2[total_pop_var2] - df2[group_pop_var2]

    df1["compl_composition"] = (df1["compl_pop_var"] / df1[total_pop_var1]).fillna(0)
    df2["compl_composition"] = (df2["compl_pop_var"] / df2[total_pop_var2]).fillna(0)

    df1["counterfactual_group_pop"] = (
        df1["group_composition"].rank(pct=True).apply(df2["group_composition"].quantile)
        * df1[total_pop_var1]
    )
    df2["counterfactual_group_pop"] = (
        df2["group_composition"].rank(pct=True).apply(df1["group_composition"].quantile)
        * df2[total_pop_var2]
    )

    df1["counterfactual_compl_pop"] = (
        df1["compl_composition"].rank(pct=True).apply(df2["compl_composition"].quantile)
        * df1[total_pop_var1]
    )
    df2["counterfactual_compl_pop"] = (
        df2["compl_composition"].rank(pct=True).apply(df1["compl_composition"].quantile)
        * df2[total_pop_var2]
    )

    df1["counterfactual_total_pop"] = (
        df1["counterfactual_group_pop"] + df1["counterfactual_compl_pop"]
    )
    df2["counterfactual_total_pop"] = (
        df2["counterfactual_group_pop"] + df2["counterfactual_compl_pop"]
    )

    return df1, df2


def sim_share(
    df1, df2, group_pop_var1, total_pop_var1, group_pop_var2, total_pop_var2,
):
    """Simulate the spatial population distribution of a region using the CDF of a comparison region.

    For each spatial unit i in region 1, take the unit's percentile in the distribution, and swap the group share
    with the value of the corresponding percentile in region 2. The share is the minority population of unit i
    divided by total population of minority population. This approach will shift the total population of
    each unit without changing the regional proportion of each group

    Parameters
    ----------
    df1 : pandas.DataFrame or geopandas.GeoDataFrame
        dataframe for first dataset with columns holding group and total population counts
    df2 : pandas.DataFrame or geopandas.GeoDataFrame
        dataframe for second dataset with columns holding group and total population counts
    group_pop_var1 : str
        column holding population counts for group of interest on input df1
    total_pop_var1 : str
        column holding total population counts on input df1
    group_pop_var2 : str
        column holding population counts for group of interest on input df2
    total_pop_var2 : str
        column holding total population counts on input df2

    Returns
    -------
    two pandas.DataFrame
        dataframes with simulated population columns appended

    """
    df1, df2 = _prepare_comparative_data(df1, df2, group_pop_var1, group_pop_var2, total_pop_var1, total_pop_var2)

    df1["compl_pop_var"] = df1[total_pop_var1] - df1[group_pop_var1]
    df2["compl_pop_var"] = df2[total_pop_var2] - df2[group_pop_var2]

    df1["share"] = (df1[group_pop_var1] / df1[group_pop_var1].sum()).fillna(0)
    df2["share"] = (df2[group_pop_var2] / df2[group_pop_var2].sum()).fillna(0)

    df1["compl_share"] = (df1["compl_pop_var"] / df1["compl_pop_var"].sum()).fillna(0)
    df2["compl_share"] = (df2["compl_pop_var"] / df2["compl_pop_var"].sum()).fillna(0)

    # Rescale due to possibility of the summation of the counterfactual share values being grater or lower than 1
    # CT stands for Correction Term
    CT1_2_group = df1["share"].rank(pct=True).apply(df2["share"].quantile).sum()
    CT2_1_group = df2["share"].rank(pct=True).apply(df1["share"].quantile).sum()

    df1["counterfactual_group_pop"] = (
        df1["share"].rank(pct=True).apply(df2["share"].quantile)
        / CT1_2_group
        * df1[group_pop_var1].sum()
    )
    df2["counterfactual_group_pop"] = (
        df2["share"].rank(pct=True).apply(df1["share"].quantile)
        / CT2_1_group
        * df2[group_pop_var2].sum()
    )

    # Rescale due to possibility of the summation of the counterfactual share values being grater or lower than 1
    # CT stands for Correction Term
    CT1_2_compl = (
        df1["compl_share"].rank(pct=True).apply(df2["compl_share"].quantile).sum()
    )
    CT2_1_compl = (
        df2["compl_share"].rank(pct=True).apply(df1["compl_share"].quantile).sum()
    )

    df1["counterfactual_compl_pop"] = (
        df1["compl_share"].rank(pct=True).apply(df2["compl_share"].quantile)
        / CT1_2_compl
        * df1["compl_pop_var"].sum()
    )
    df2["counterfactual_compl_pop"] = (
        df2["compl_share"].rank(pct=True).apply(df1["compl_share"].quantile)
        / CT2_1_compl
        * df2["compl_pop_var"].sum()
    )

    df1["counterfactual_total_pop"] = (
        df1["counterfactual_group_pop"] + df1["counterfactual_compl_pop"]
    )
    df2["counterfactual_total_pop"] = (
        df2["counterfactual_group_pop"] + df2["counterfactual_compl_pop"]
    )
    return df1.fillna(0), df2.fillna(0)


def _prepare_random_label(seg_class_1, seg_class_2):
    if hasattr(seg_class_1, "_original_data"):
        data_1 = seg_class_1._original_data.copy()
    else:
        data_1 = seg_class_1.data.copy()
    if hasattr(seg_class_2, "_original_data"):
        data_2 = seg_class_2._original_data.copy()
    else:
        data_2 = seg_class_2.data.copy()

    data_1["grouping_variable"] = "Group_1"
    data_2["grouping_variable"] = "Group_2"

    if isinstance(seg_class_1, SingleGroupIndex):

        # This step is just to make sure the each frequency column is integer for the approaches and from the same type in order to be able to stack them
        data_1.loc[:, (seg_class_1.group_pop_var, seg_class_1.total_pop_var)] = (
            data_1.loc[:, (seg_class_1.group_pop_var, seg_class_1.total_pop_var)]
            .round(0)
            .astype(int)
        )

        # random permutation needs the columns to have the same names
        data_1 = data_1[
            [seg_class_1.group_pop_var, seg_class_1.total_pop_var, "grouping_variable",]
        ]
        data_1.columns = ["group", "total", "grouping_variable"]

        data_2.loc[:, (seg_class_2.group_pop_var, seg_class_2.total_pop_var)] = (
            data_2.loc[:, (seg_class_2.group_pop_var, seg_class_2.total_pop_var)]
            .round(0)
            .astype(int)
        )
        data_2 = data_2[
            [seg_class_2.group_pop_var, seg_class_2.total_pop_var, "grouping_variable",]
        ]
        data_2.columns = ["group", "total", "grouping_variable"]

        stacked_data = pd.concat([data_1, data_2], axis=0)

    elif isinstance(seg_class_1, MultiGroupIndex):

        groups_list = seg_class_1.groups

        for i in range(len(groups_list)):
            data_1[groups_list[i]] = round(data_1[groups_list[i]]).astype(int)
            data_2[groups_list[i]] = round(data_2[groups_list[i]]).astype(int)

        if seg_class_1.groups != seg_class_2.groups:
            raise ValueError("MultiGroup groups should be the same")

        stacked_data = pd.concat([data_1, data_2], ignore_index=True)
    return stacked_data


def _estimate_random_label_difference(data):
    #  note: if estimating a spatial implicit index, then "space" has already been accounted for...
    #  when the index is computed, the underlying data are transformed to represent the *accessible* population
    #  so when calculating the simulated difference, we need to pop spatial implicit parameters
    
    stacked_data = data[0]
    function = data[1]
    index_args_1 = data[2]
    index_args_2 = data[3]
    idx_type = data[4]
    groups = data[5]
    approach = data[6]
    for args in [index_args_1, index_args_2]:
        if 'network' in args:
            args.pop('network')
        elif 'distance' in args:
            args.pop('distance')

    if approach == 'person_permutation':
        grouping = stacked_data['grouping_variable'].copy().values
        if groups:
            stacked_data = simulate_person_permutation(stacked_data, groups=groups)
        else:
            stacked_data = simulate_person_permutation(stacked_data, group='group', total='total')
        stacked_data['grouping_variable'] = grouping

    else:
        stacked_data["grouping_variable"] = np.random.permutation(
            stacked_data["grouping_variable"].values
        )

    stacked_data_1 = stacked_data[stacked_data["grouping_variable"] == "Group_1"]
    stacked_data_2 = stacked_data[stacked_data["grouping_variable"] == "Group_2"]
    if idx_type == "singlegroup":
        simulations_1 = function(stacked_data_1, "group", "total", **index_args_1)[0]
        simulations_2 = function(stacked_data_2, "group", "total", **index_args_2)[0]
    elif idx_type == "multigroup":
        simulations_1 = function(stacked_data_1, groups, **index_args_1)[0]
        simulations_2 = function(stacked_data_2, groups, **index_args_2)[0]

    est = simulations_1 - simulations_2

    return est


def _estimate_counterfac_difference(data):
    data_1 = data[0]
    data_2 = data[1]
    counterfac_df1 = data[10]
    counterfac_df2 = data[11]

    group_1 = data[2]
    total_1 = data[3]
    group_2 = data[4]
    total_2 = data[5]
    index_args_1 = data[6]
    index_args_2 = data[7]
    approach = data[8]
    function = data[9]

    if approach in ["counterfactual_share", "counterfactual_dual_composition"]:
        data_1[total_1] = counterfac_df1["counterfactual_total_pop"]
        data_2[total_2] = counterfac_df2["counterfactual_total_pop"]

    data_1["fair_coin"] = np.random.uniform(size=len(data_1))
    data_1["test_group_pop_var"] = np.where(
        data_1["fair_coin"] > 0.5,
        data_1[group_1],
        counterfac_df1["counterfactual_group_pop"],
    )

    # Dropping to avoid confusion in the internal function
    data_1_test = data_1.drop([group_1], axis=1)

    simulations_1 = function(
        data_1_test, "test_group_pop_var", total_1, **index_args_1,
    )[0]

    # Dropping to avoid confusion in the next iteration
    data_1 = data_1.drop(["fair_coin", "test_group_pop_var"], axis=1)

    data_2["fair_coin"] = np.random.uniform(size=len(data_2))
    data_2["test_group_pop_var"] = np.where(
        data_2["fair_coin"] > 0.5,
        data_2[group_2],
        counterfac_df2["counterfactual_group_pop"],
    )

    # Dropping to avoid confusion in the internal function
    data_2_test = data_2.drop([group_2], axis=1)

    simulations_2 = function(
        data_2_test, "test_group_pop_var", total_2, **index_args_2,
    )[0]

    # Dropping to avoid confusion in the next iteration
    data_2 = data_2.drop(["fair_coin", "test_group_pop_var"], axis=1)

    est = simulations_1 - simulations_2

    return est


DUAL_SIMULATORS = {
    "composition": sim_composition,
    "dual_composition": sim_dual_composition,
    "share": sim_share,
}
