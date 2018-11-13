"""
Useful functions for segregation metrics
"""

__author__ = "Levi Wolf <levi.john.wolf@gmail.com> and Renan X. Cortes <renanc@ucr.edu>"

import numpy as np
import pandas as pd
import libpysal
import geopandas as gpd
from warnings import warn
from libpysal.weights import Queen, KNN
from libpysal.weights.util import attach_islands


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
    
    w = libpysal.weights.Rook.from_dataframe(data, ids = data.index.tolist(),
                                              geom_col=data._geometry_column_name)
    
    if not len(w.islands):
        w = w
    else:
        warn('There are some islands in the GeoDataFrame.')
    
    adjlist = w.to_adjlist()
    merged = adjlist.merge(data.geometry.to_frame('geometry'), left_on='focal',
                           right_index=True, how='left')\
                    .merge(data.geometry.to_frame('geometry'), left_on='neighbor',
                           right_index=True, how='left', suffixes=("_focal", "_neighbor"))\
    
    # Transforming from pandas to geopandas
    merged = gpd.GeoDataFrame(merged, geometry='geometry_focal')
    merged['geometry_neighbor'] = gpd.GeoSeries(merged.geometry_neighbor)
    
    # Getting the shared boundaries
    merged['shared_boundary'] = merged.geometry_focal.intersection(merged.set_geometry('geometry_neighbor'))
    
    # Putting it back to a matrix
    merged['weight'] = merged.set_geometry('shared_boundary').length
    merged_with_islands = pd.concat((merged, islands))
    length_weighted_w = libpysal.weights.W.from_adjlist(merged_with_islands[['focal', 'neighbor', 'weight']])
    neighbors, weights = length_weighted_w.neighbors, length_weighted_w.weights
    for island in w.islands:
        length_weighted_w.neighbors[island] = []
        del length_weighted_w.weights[island]
    
    length_weighted_w._reset()
    
    return length_weighted_w
