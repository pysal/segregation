"""
Useful functions for segregation metrics
"""

__author__ = "Levi Wolf <levi.john.wolf@gmail.com>, Renan X. Cortes <renanc@ucr.edu>, and Eli Knaap <ek@knaaptime.com>"

import numpy as np
import pandas as pd
import libpysal
import geopandas as gpd
import math
from warnings import warn
from libpysal.weights import Kernel
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


def project_gdf(gdf, to_crs=None, to_latlong=False):
    """Reproject gdf into the appropriate UTM zone.

    Project a GeoDataFrame to the UTM zone appropriate for its geometries'
    centroid.
    The simple calculation in this function works well for most latitudes, but
    won't work for some far northern locations like Svalbard and parts of far
    northern Norway.

    This function is lovingly modified from osmnx:
    https://github.com/gboeing/osmnx/

    Parameters
    ----------
    gdf : GeoDataFrame
        the gdf to be projected
    to_crs : dict
        if not None, just project to this CRS instead of to UTM
    to_latlong : bool
        if True, projects to latlong instead of to UTM

    Returns
    -------
    GeoDataFrame

    """
    assert len(gdf) > 0, 'You cannot project an empty GeoDataFrame.'

    # else, project the gdf to UTM
    # if GeoDataFrame is already in UTM, just return it
    if (gdf.crs is not None) and ('+proj=utm ' in gdf.crs):
        return gdf

    # calculate the centroid of the union of all the geometries in the
    # GeoDataFrame
    avg_longitude = gdf['geometry'].unary_union.centroid.x

    # calculate the UTM zone from this avg longitude and define the UTM
    # CRS to project
    utm_zone = int(math.floor((avg_longitude + 180) / 6.) + 1)
    utm_crs = '+proj=utm +zone={} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'.format(
        utm_zone)

    # project the GeoDataFrame to the UTM CRS
    projected_gdf = gdf.to_crs(utm_crs)

    return projected_gdf


def compute_segregation_profile(gdf,
                                groups=None,
                                distances=None,
                                network=None,
                                decay='linear',
                                function='triangular',
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
        list of floats representing bandwidth distances that define a local
        environment.
    network : pandana.Network (optional)
        A pandana.Network likely created with
        `segregation.network.get_network`.
    decay : str (optional)
        decay type to be used in pandana accessibility calculation (the
        default is 'linear').
    function: 'str' (optional)
        which weighting function should be passed to libpysal.weights.Kernel
        must be one of: 'triangular','uniform','quadratic','quartic','gaussian'
    precompute: bool
        Whether the pandana.Network instance should precompute the range
        queries.This is true by default, but if you plan to calculate several
        segregation profiles using the same network, then you can set this
        parameter to `False` to avoid precomputing repeatedly inside the
        function

    Returns
    -------
    dict
        dictionary with distances as keys and SIT statistics as values

    Notes
    -----
    Based on Sean F. Reardon, Stephen A. Matthews, David O’Sullivan, Barrett A. Lee, Glenn Firebaugh, Chad R. Farrell, & Kendra Bischoff. (2008). The Geographic Scale of Metropolitan Racial Segregation. Demography, 45(3), 489–514. https://doi.org/10.1353/dem.0.0019.

    Reference: :cite:`Reardon2008`.

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
            w = Kernel.from_dataframe(gdf,
                                      bandwidth=distance,
                                      function=function)
            sit = SpatialInformationTheory(gdf, groups, w=w)
            indices[distance] = sit.statistic
    return indices
