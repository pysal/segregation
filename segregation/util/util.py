"""
Useful functions for segregation metrics
"""

__author__ = "Levi Wolf <levi.john.wolf@gmail.com>, Renan X. Cortes <renanc@ucr.edu>, and Eli Knaap <ek@knaaptime.com>"


import numpy as np
import math
import warnings
from pyproj import CRS


def _nan_handle(df):
    """Check if dataframe has nan values.
    Raise an informative error.
    """
    if str(type(df)) == "<class 'geopandas.geodataframe.GeoDataFrame'>":
        values = df.loc[:, df.columns != df._geometry_column_name].values
    else:
        values = df.values

    if np.any(np.isnan(values)):
        warnings.warn(
            "There are NAs present in the input data. NAs should be handled (e.g. dropping or replacing them with values) before using this function."
        )

    return df


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
    if (group_pop_var1 not in data1.columns) or (total_pop_var1 not in data1.columns):
        raise ValueError("group_pop_var and total_pop_var must be variables of data1")

    if (group_pop_var2 not in data2.columns) or (total_pop_var2 not in data2.columns):
        raise ValueError("group_pop_var and total_pop_var must be variables of data2")

    if any(data1[total_pop_var1] < data1[group_pop_var1]):
        raise ValueError(
            "Group of interest population must equal or lower than the total population of the units in data1."
        )

    if any(data2[total_pop_var2] < data2[group_pop_var2]):
        raise ValueError(
            "Group of interest population must equal or lower than the total population of the units in data2."
        )

    df1 = data1.copy()
    df2 = data2.copy()

    if counterfactual_approach == "composition":

        df1["group_composition"] = np.where(
            df1[total_pop_var1] == 0, 0, df1[group_pop_var1] / df1[total_pop_var1]
        )
        df2["group_composition"] = np.where(
            df2[total_pop_var2] == 0, 0, df2[group_pop_var2] / df2[total_pop_var2]
        )

        df1["counterfactual_group_pop"] = (
            df1["group_composition"]
            .rank(pct=True)
            .apply(df2["group_composition"].quantile)
            * df1[total_pop_var1]
        )
        df2["counterfactual_group_pop"] = (
            df2["group_composition"]
            .rank(pct=True)
            .apply(df1["group_composition"].quantile)
            * df2[total_pop_var2]
        )

        df1["counterfactual_total_pop"] = df1[total_pop_var1]
        df2["counterfactual_total_pop"] = df2[total_pop_var2]

    if counterfactual_approach == "share":

        df1["compl_pop_var"] = df1[total_pop_var1] - df1[group_pop_var1]
        df2["compl_pop_var"] = df2[total_pop_var2] - df2[group_pop_var2]

        df1["share"] = np.where(
            df1[total_pop_var1] == 0, 0, df1[group_pop_var1] / df1[group_pop_var1].sum()
        )
        df2["share"] = np.where(
            df2[total_pop_var2] == 0, 0, df2[group_pop_var2] / df2[group_pop_var2].sum()
        )

        df1["compl_share"] = np.where(
            df1["compl_pop_var"] == 0,
            0,
            df1["compl_pop_var"] / df1["compl_pop_var"].sum(),
        )
        df2["compl_share"] = np.where(
            df2["compl_pop_var"] == 0,
            0,
            df2["compl_pop_var"] / df2["compl_pop_var"].sum(),
        )

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

    if counterfactual_approach == "dual_composition":

        df1["group_composition"] = np.where(
            df1[total_pop_var1] == 0, 0, df1[group_pop_var1] / df1[total_pop_var1]
        )
        df2["group_composition"] = np.where(
            df2[total_pop_var2] == 0, 0, df2[group_pop_var2] / df2[total_pop_var2]
        )

        df1["compl_pop_var"] = df1[total_pop_var1] - df1[group_pop_var1]
        df2["compl_pop_var"] = df2[total_pop_var2] - df2[group_pop_var2]

        df1["compl_composition"] = np.where(
            df1[total_pop_var1] == 0, 0, df1["compl_pop_var"] / df1[total_pop_var1]
        )
        df2["compl_composition"] = np.where(
            df2[total_pop_var2] == 0, 0, df2["compl_pop_var"] / df2[total_pop_var2]
        )

        df1["counterfactual_group_pop"] = (
            df1["group_composition"]
            .rank(pct=True)
            .apply(df2["group_composition"].quantile)
            * df1[total_pop_var1]
        )
        df2["counterfactual_group_pop"] = (
            df2["group_composition"]
            .rank(pct=True)
            .apply(df1["group_composition"].quantile)
            * df2[total_pop_var2]
        )

        df1["counterfactual_compl_pop"] = (
            df1["compl_composition"]
            .rank(pct=True)
            .apply(df2["compl_composition"].quantile)
            * df1[total_pop_var1]
        )
        df2["counterfactual_compl_pop"] = (
            df2["compl_composition"]
            .rank(pct=True)
            .apply(df1["compl_composition"].quantile)
            * df2[total_pop_var2]
        )

        df1["counterfactual_total_pop"] = (
            df1["counterfactual_group_pop"] + df1["counterfactual_compl_pop"]
        )
        df2["counterfactual_total_pop"] = (
            df2["counterfactual_group_pop"] + df2["counterfactual_compl_pop"]
        )

    df1["group_composition"] = np.where(
        df1[total_pop_var1] == 0, 0, df1[group_pop_var1] / df1[total_pop_var1]
    )
    df2["group_composition"] = np.where(
        df2[total_pop_var2] == 0, 0, df2[group_pop_var2] / df2[total_pop_var2]
    )

    df1["counterfactual_composition"] = np.where(
        df1["counterfactual_total_pop"] == 0,
        0,
        df1["counterfactual_group_pop"] / df1["counterfactual_total_pop"],
    )
    df2["counterfactual_composition"] = np.where(
        df2["counterfactual_total_pop"] == 0,
        0,
        df2["counterfactual_group_pop"] / df2["counterfactual_total_pop"],
    )

    df1 = df1.drop(columns=[group_pop_var1, total_pop_var1], axis=1)
    df2 = df2.drop(columns=[group_pop_var2, total_pop_var2], axis=1)

    return df1, df2

