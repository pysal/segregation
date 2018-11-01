"""
Useful functions for segregation metrics
"""

__author__ = "Levi Wolf <levi.john.wolf@bristol.ac.uk> and Renan X. Cortes <renanc@ucr.edu>"

import numpy as np
import pandas as pd
import libpysal
import geopandas as gpd


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
    
    if ('geometry' not in data.columns):    
        raise ValueError('The input data has to have a column named \'geometry\'')
    
    data['index'] = data.index
    w = libpysal.weights.Queen.from_dataframe(data, ids = data.index.tolist())
    
    adjlist = w.to_adjlist().merge(data[['index', 'geometry']], left_on='focal', right_on='index', how='left')\
              .drop('index', axis=1)\
              .merge(data[['index', 'geometry']], left_on='neighbor', right_on='index', 
                     how='left', suffixes=("_focal", "_neighbor"))\
              .drop('index', axis=1)
    
    # Transforming from pandas to geopandas
    adjlist = gpd.GeoDataFrame(adjlist, geometry='geometry_focal')
    adjlist['geometry_neighbor'] = gpd.GeoSeries(adjlist.geometry_neighbor)
    
    # Getting the shared boundaries
    adjlist['shared_boundary'] = adjlist.geometry_focal.intersection(adjlist.set_geometry('geometry_neighbor'))
    
    # Putting it back to a matrix
    adjlist['weight'] = adjlist.set_geometry('shared_boundary').length
    length_weighted_w = libpysal.weights.W.from_adjlist(adjlist[['focal', 'neighbor', 'weight']])
    
    return length_weighted_w
