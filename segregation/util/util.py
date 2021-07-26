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


def _dep_message(original, replacement, when="2020-01-31", version="2.1.0"):
    msg = "Deprecated (%s): %s" % (version, original)
    msg += " is being renamed to %s." % replacement
    msg += " %s will be removed on %s." % (original, when)
    return msg


class DeprecationHelper(object):
    def __init__(self, new_target, message="Deprecated"):
        self.new_target = new_target
        self.message = message

    def _warn(self):
        from warnings import warn

        warn(self.message)

    def __call__(self, *args, **kwargs):
        self._warn()
        return self.new_target(*args, **kwargs)

    def __getattr__(self, attr):
        self._warn()
        return getattr(self.new_target, attr)
