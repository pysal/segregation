"""
Useful functions for segregation metrics
"""

__author__ = "Levi Wolf <levi.john.wolf@gmail.com>, Renan X. Cortes <renanc@ucr.edu>, and Eli Knaap <ek@knaaptime.com>"


import numpy as np
import pandas as pd
import warnings
from sklearn.metrics.pairwise import euclidean_distances


def generate_distance_matrix(data):
    """Generate a pairwise Euclidean distance matrix from the rows in a geodataframe.

    Parameters
    ----------
    data : geopandas.GeoDataFrame
        A geodataframe with a projected coordinate system

    Returns
    -------
    numpy.array
        distance matrix
    """
    assert data.crs.is_projected, "This function can only operate on euclidean coordinates. Please reproject the geodataframe"
    w = euclidean_distances(
        pd.DataFrame({"x": data.centroid.x.values, "y": data.centroid.y.values})
    )
    w = w / w.sum(axis=1)
    dist = np.exp(-w)
    return dist


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
