"""Compute multiscalar segregation profiles."""

import warnings

import numpy as np
import pandas as pd
from libpysal.weights import Kernel
from pyproj.crs import CRS


def compute_multiscalar_profile(
    gdf,
    segregation_index=None,
    groups=None,
    group_pop_var=None,
    total_pop_var=None,
    distances=None,
    network=None,
    decay="linear",
    function="triangular",
    precompute=True,
    **kwargs
):
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
    segregation_index : SpatialImplicit SegregationIndex Class
        a class from the library such as MultiInformationTheory, or MinMax
    groups : list
        list of population groups for calculating multigroup indices
    group_pop_var : str
        name of population group on gdf for calculating single group indices
    total_pop_var : str
        bame of total population on gdf for calculating single group indices
    distances : list
        list of floats representing bandwidth distances that define a local
        environment.
    network : pandana.Network (optional)
        A pandana.Network likely created with
        `segregation.network.get_osm_network`.
    decay : str (optional)
        decay type to be used in pandana accessibility calculation
        options are {'linear', 'exp', 'flat'}. The default is 'linear'.
    function: 'str' (optional)
        which weighting function should be passed to libpysal.weights.Kernel
        must be one of: 'triangular','uniform','quadratic','quartic','gaussian'
    precompute: bool
        Whether the pandana.Network instance should precompute the range
        queries. This is True by default
    **kwargs : dict
        additional keyword arguments passed to each index (e.g. for setting a random
        seed in indices like ModifiedGini or ModifiedDissm)


    Returns
    -------
    pandas.Series
        Series with distances as index and index statistics as values

    Notes
    -----
    Based on Sean F. Reardon, Stephen A. Matthews, David O’Sullivan, Barrett A. Lee, Glenn Firebaugh, Chad R. Farrell, & Kendra Bischoff. (2008). The Geographic Scale of Metropolitan Racial Segregation. Demography, 45(3), 489–514. https://doi.org/10.1353/dem.0.0019.

    Reference: :cite:`reardon2008`.

    """
    if not segregation_index:
        raise ValueError("You must pass a segregation SpatialImplicit Index Class")
    gdf = gdf.copy()
    indices = {}

    if groups:
        gdf[groups] = gdf[groups].astype(float)
        indices[0] = segregation_index(gdf, groups=groups, **kwargs).statistic
    elif group_pop_var:
        indices[0] = segregation_index(
            gdf, group_pop_var=group_pop_var, total_pop_var=total_pop_var, **kwargs
        ).statistic

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if network:
            if not gdf.crs.equals(CRS(4326)):
                gdf = gdf.to_crs(epsg=4326)
            if precompute:
                maxdist = max(distances)
                network.precompute(maxdist)
            for distance in distances:
                distance = np.float(distance)
                if group_pop_var:
                    idx = segregation_index(
                        gdf,
                        group_pop_var=group_pop_var,
                        total_pop_var=total_pop_var,
                        network=network,
                        decay=decay,
                        distance=distance,
                        precompute=False,
                        **kwargs
                    )
                elif groups:
                    idx = segregation_index(
                        gdf,
                        groups=groups,
                        network=network,
                        decay=decay,
                        distance=distance,
                        precompute=False,
                        **kwargs
                    )

                indices[distance] = idx.statistic
        else:
            for distance in distances:
                w = Kernel.from_dataframe(gdf, bandwidth=distance, function=function)
                if group_pop_var:
                    idx = segregation_index(
                        gdf,
                        group_pop_var=group_pop_var,
                        total_pop_var=total_pop_var,
                        w=w,
                        **kwargs
                    )
                else:
                    idx = segregation_index(gdf, groups, w=w, **kwargs)
                indices[distance] = idx.statistic
        series = pd.Series(indices, name=str(type(idx)).split(".")[-1][:-2])
        series.index.name = "distance"
        return series
