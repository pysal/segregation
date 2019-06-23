"""
Useful functions for segregation metrics
"""

__author__ = "Levi Wolf <levi.john.wolf@gmail.com> and Renan X. Cortes <renanc@ucr.edu>"

import numpy as np
import pandas as pd
import libpysal
import geopandas as gpd
from warnings import warn
from libpysal.weights import Queen, KNN, Kernel
from libpysal.weights.util import attach_islands
from segregation.network import calc_access
from segregation.spatial import SpatialInformationTheory
from segregation.aspatial import Multi_Information_Theory


def _return_length_weighted_w(data):
    """
    Returns a PySAL weights object that the weights represent the length of the commom boudary of two areal units that share border.

    Parameters
    ----------

    data          : a geopandas DataFrame with a 'geometry' column.

    Notes
    -----
    Currently it's not making any projection.

    """

    w = libpysal.weights.Rook.from_dataframe(
        data, ids=data.index.tolist(), geom_col=data._geometry_column_name)

    if (len(w.islands) == 0):
        w = w
    else:
        warn('There are some islands in the GeoDataFrame.')
        w_aux = libpysal.weights.KNN.from_dataframe(
            data,
            ids=data.index.tolist(),
            geom_col=data._geometry_column_name,
            k=1)
        w = attach_islands(w, w_aux)

    adjlist = w.to_adjlist()
    islands = pd.DataFrame.from_records([{
        'focal': island,
        'neighbor': island,
        'weight': 0
    } for island in w.islands])
    merged = adjlist.merge(data.geometry.to_frame('geometry'), left_on='focal',
                           right_index=True, how='left')\
                    .merge(data.geometry.to_frame('geometry'), left_on='neighbor',
                           right_index=True, how='left', suffixes=("_focal", "_neighbor"))\

    # Transforming from pandas to geopandas
    merged = gpd.GeoDataFrame(merged, geometry='geometry_focal')
    merged['geometry_neighbor'] = gpd.GeoSeries(merged.geometry_neighbor)

    # Getting the shared boundaries
    merged['shared_boundary'] = merged.geometry_focal.intersection(
        merged.set_geometry('geometry_neighbor'))

    # Putting it back to a matrix
    merged['weight'] = merged.set_geometry('shared_boundary').length
    merged_with_islands = pd.concat((merged, islands))
    length_weighted_w = libpysal.weights.W.from_adjlist(
        merged_with_islands[['focal', 'neighbor', 'weight']])
    for island in w.islands:
        length_weighted_w.neighbors[island] = []
        del length_weighted_w.weights[island]

    length_weighted_w._reset()

    return length_weighted_w


def _generate_counterfactual(data1,
                             data2,
                             group_pop_var,
                             total_pop_var,
                             counterfactual_approach='composition'):
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
    if ((type(group_pop_var) is not str) or (type(total_pop_var) is not str)):
        raise TypeError('group_pop_var and total_pop_var must be strings')

    if ((group_pop_var not in data1.columns)
            or (total_pop_var not in data1.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data1')

    if ((group_pop_var not in data2.columns)
            or (total_pop_var not in data2.columns)):
        raise ValueError(
            'group_pop_var and total_pop_var must be variables of data2')

    if any(data1[total_pop_var] < data1[group_pop_var]):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units in data1.'
        )

    if any(data2[total_pop_var] < data2[group_pop_var]):
        raise ValueError(
            'Group of interest population must equal or lower than the total population of the units in data2.'
        )

    df1 = data1.copy()
    df2 = data2.copy()

    if (counterfactual_approach == 'composition'):

        df1['group_composition'] = np.where(
            df1[total_pop_var] == 0, 0,
            df1[group_pop_var] / df1[total_pop_var])
        df2['group_composition'] = np.where(
            df2[total_pop_var] == 0, 0,
            df2[group_pop_var] / df2[total_pop_var])

        df1['counterfactual_group_pop'] = df1['group_composition'].rank(
            pct=True).apply(
                df2['group_composition'].quantile) * df1[total_pop_var]
        df2['counterfactual_group_pop'] = df2['group_composition'].rank(
            pct=True).apply(
                df1['group_composition'].quantile) * df2[total_pop_var]

        df1['counterfactual_total_pop'] = df1[total_pop_var]
        df2['counterfactual_total_pop'] = df2[total_pop_var]

    if (counterfactual_approach == 'share'):

        df1['compl_pop_var'] = df1[total_pop_var] - df1[group_pop_var]
        df2['compl_pop_var'] = df2[total_pop_var] - df2[group_pop_var]

        df1['share'] = np.where(df1[total_pop_var] == 0, 0,
                                df1[group_pop_var] / df1[group_pop_var].sum())
        df2['share'] = np.where(df2[total_pop_var] == 0, 0,
                                df2[group_pop_var] / df2[group_pop_var].sum())

        df1['compl_share'] = np.where(
            df1['compl_pop_var'] == 0, 0,
            df1['compl_pop_var'] / df1['compl_pop_var'].sum())
        df2['compl_share'] = np.where(
            df2['compl_pop_var'] == 0, 0,
            df2['compl_pop_var'] / df2['compl_pop_var'].sum())

        # Rescale due to possibility of the summation of the counterfactual share values being grater or lower than 1
        # CT stands for Correction Term
        CT1_2_group = df1['share'].rank(pct=True).apply(
            df2['share'].quantile).sum()
        CT2_1_group = df2['share'].rank(pct=True).apply(
            df1['share'].quantile).sum()

        df1['counterfactual_group_pop'] = df1['share'].rank(pct=True).apply(
            df2['share'].quantile) / CT1_2_group * df1[group_pop_var].sum()
        df2['counterfactual_group_pop'] = df2['share'].rank(pct=True).apply(
            df1['share'].quantile) / CT2_1_group * df2[group_pop_var].sum()

        # Rescale due to possibility of the summation of the counterfactual share values being grater or lower than 1
        # CT stands for Correction Term
        CT1_2_compl = df1['compl_share'].rank(pct=True).apply(
            df2['compl_share'].quantile).sum()
        CT2_1_compl = df2['compl_share'].rank(pct=True).apply(
            df1['compl_share'].quantile).sum()

        df1['counterfactual_compl_pop'] = df1['compl_share'].rank(
            pct=True).apply(df2['compl_share'].quantile
                            ) / CT1_2_compl * df1['compl_pop_var'].sum()
        df2['counterfactual_compl_pop'] = df2['compl_share'].rank(
            pct=True).apply(df1['compl_share'].quantile
                            ) / CT2_1_compl * df2['compl_pop_var'].sum()

        df1['counterfactual_total_pop'] = df1[
            'counterfactual_group_pop'] + df1['counterfactual_compl_pop']
        df2['counterfactual_total_pop'] = df2[
            'counterfactual_group_pop'] + df2['counterfactual_compl_pop']

    if (counterfactual_approach == 'dual_composition'):

        df1['group_composition'] = np.where(
            df1[total_pop_var] == 0, 0,
            df1[group_pop_var] / df1[total_pop_var])
        df2['group_composition'] = np.where(
            df2[total_pop_var] == 0, 0,
            df2[group_pop_var] / df2[total_pop_var])

        df1['compl_pop_var'] = df1[total_pop_var] - df1[group_pop_var]
        df2['compl_pop_var'] = df2[total_pop_var] - df2[group_pop_var]

        df1['compl_composition'] = np.where(
            df1[total_pop_var] == 0, 0,
            df1['compl_pop_var'] / df1[total_pop_var])
        df2['compl_composition'] = np.where(
            df2[total_pop_var] == 0, 0,
            df2['compl_pop_var'] / df2[total_pop_var])

        df1['counterfactual_group_pop'] = df1['group_composition'].rank(
            pct=True).apply(
                df2['group_composition'].quantile) * df1[total_pop_var]
        df2['counterfactual_group_pop'] = df2['group_composition'].rank(
            pct=True).apply(
                df1['group_composition'].quantile) * df2[total_pop_var]

        df1['counterfactual_compl_pop'] = df1['compl_composition'].rank(
            pct=True).apply(
                df2['compl_composition'].quantile) * df1[total_pop_var]
        df2['counterfactual_compl_pop'] = df2['compl_composition'].rank(
            pct=True).apply(
                df1['compl_composition'].quantile) * df2[total_pop_var]

        df1['counterfactual_total_pop'] = df1[
            'counterfactual_group_pop'] + df1['counterfactual_compl_pop']
        df2['counterfactual_total_pop'] = df2[
            'counterfactual_group_pop'] + df2['counterfactual_compl_pop']

    df1['group_composition'] = np.where(
        df1['total_pop_var'] == 0, 0,
        df1['group_pop_var'] / df1['total_pop_var'])
    df2['group_composition'] = np.where(
        df2['total_pop_var'] == 0, 0,
        df2['group_pop_var'] / df2['total_pop_var'])

    df1['counterfactual_composition'] = np.where(
        df1['counterfactual_total_pop'] == 0, 0,
        df1['counterfactual_group_pop'] / df1['counterfactual_total_pop'])
    df2['counterfactual_composition'] = np.where(
        df2['counterfactual_total_pop'] == 0, 0,
        df2['counterfactual_group_pop'] / df2['counterfactual_total_pop'])

    df1 = df1.drop(columns=['group_pop_var', 'total_pop_var'], axis=1)
    df2 = df2.drop(columns=['group_pop_var', 'total_pop_var'], axis=1)

    return df1, df2


def compute_segregation_profile(gdf,
                                groups=None,
                                distances=None,
                                network=None,
                                decay='linear',
                                precompute=True):
    """Compute multiscalar segregation profile.

    This function calculates several Spatial Information Theory indices with
    increasing distance parameters.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        geodataframe with rows as observations and columns as population
        variables. Note that if using a network distance, the coordinate
        system for this gdf should be 4326. If using euclidian distance,
        this must be projected into planar coordinates like state plane or UTM.
    groups : list
        list of variables .
    distances : list
        list of integers representing bandwidth distances that define a local
        environment.
    network : pandana.Network (optional)
        A pandana.Network likely created with
        `segregation.network.get_network`.
    decay : str (optional)
        decay type to be used in pandana accessibility calculation (the
        default is 'linear').

    Returns
    -------
    dict
        dictionary with distances as keys and SIT statistics as values

    """
    gdf = gdf.copy()
    gdf[groups] = gdf[groups].astype(float)
    indices = {}
    indices[0] = Multi_Information_Theory(gdf, groups).statistic

    if network:
        if not gdf.crs['init'] == 'epsg:4326':
            gdf = gdf.to_crs(epsg=4326)
        groups2 = ['acc_' + group for group in groups]
        if precompute:
            maxdist = max(distances)
            network.precompute(maxdist)
        for distance in distances:
            distance = np.float(distance)
            access = calc_access(gdf,
                                 network,
                                 decay=decay,
                                 variables=groups,
                                 distance=distance,
                                 precompute=False)
            sit = Multi_Information_Theory(access, groups2)
            indices[distance] = sit.statistic
    else:
        for distance in distances:
            w = Kernel.from_dataframe(gdf, bandwidth=distance)
            sit = SpatialInformationTheory(gdf, groups, w=w)
            indices[distance] = sit.statistic
    return indices
