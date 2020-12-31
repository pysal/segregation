"""Base classes for segregation indices."""

import geopandas as gpd
import pandas as pd
from libpysal.weights import lag_spatial
from libpysal.weights.distance import Kernel
from libpysal.weights.util import fill_diagonal

from .util.util import _nan_handle


class SingleGroupIndex:
    """Class for estimating single-group segregation indices."""

    def __init__(
        self,
        data,
        group_pop_var,
        total_pop_var,
    ):
        """Initialize singlegroup index.

        Parameters
        ----------
        data : pandas.DataFrame or geopandas.GeoDataFrame
            dataframe or geodataframe if spatial index holding data for location of interest
        group_pop_var : str
            name of column on dataframe holding population totals for focal group
        total_pop_var : str
            name of column on dataframe holding total overall population
        """
        data = data.copy()

        if any([type(group_pop_var) is not str, type(total_pop_var) is not str]):
            raise TypeError("group_pop_var and total_pop_var must be strings")

        if any([group_pop_var not in data.columns, total_pop_var not in data.columns]):
            raise ValueError(
                "group_pop_var and total_pop_var must be variables of data"
            )

        if any(data[total_pop_var] < data[group_pop_var]):
            raise ValueError(
                "Group of interest population must equal or lower than the total population of the units."
            )

        if isinstance(data, gpd.GeoDataFrame):
            data = _nan_handle(
                data[[group_pop_var, total_pop_var, data._geometry_column_name]]
            )
            data = data[[group_pop_var, total_pop_var, data.geometry.name]]

        else:
            data = _nan_handle(data[[group_pop_var, total_pop_var]])
            data = data[[group_pop_var, total_pop_var]]

        data["group_2_pop_var"] = data[total_pop_var] - data[group_pop_var]

        self.index_type = "singlegroup"
        self.data = data
        self.group_pop_var = group_pop_var
        self.total_pop_var = total_pop_var


class MultiGroupIndex:
    """Class for estimating multi-group segregation indices."""

    def __init__(
        self,
        data,
        groups,
    ):
        """Initialize multi-group index.

        Parameters
        ----------
        data : pandas.DataFrame or geopandas.GeoDataFrame
            dataframe or geodataframe if spatial index holding data for location of interest
        groups : list-like
            list of column names on input DataFrame that hold population totals for groups of interest
        """
        self.index_type = "multigroup"
        self.data = data
        self.groups = groups


class SpatialExplicitIndex:
    """Class for estimating segregation indices that are explicitly spatial (have no aspatial version)."""

    def __init__(self):
        """Initialize spatially explicit index."""
        self.spatial_type = "explicit"


class SpatialImplicitIndex:
    """Class for estimating segregation indices that can be spatial or aspatial."""

    def __init__(self, w, network, distance, decay, precompute):
        """Initialize spatially implicit index.

        Parameters
        ----------
        w : libpysal.weights object
            lipysal spatial kernel weights object used to define an egohood
        network : pandana.Network
            pandana Network object representing the study area
        distance : int
            Maximum distance (in units of geodataframe CRS) to consider the extent of the egohood
        decay : str
            type of decay function to apply. Options include
        precompute : bool
            Whether to precompute the pandana Network object
        """
        self.spatial_type = "implicit"
        if self.index_type == "multigroup":
            self._groups = self.groups
        elif self.index_type == "singlegroup":
            self._groups = [self.group_pop_var, self.total_pop_var, "group_2_pop_var"]
        self.original_data = self.data.copy()

        if w and network:
            raise UserException(
                "must pass either a pandana network or a pysal weights object\
                 but not both"
            )
        if network:
            df = calc_access(
                self.data,
                variables=self.groups,
                network=network,
                distance=distance,
                decay=decay,
                precompute=precompute,
            )
            self._groups = ["acc_" + group for group in self.__groups]
            self.network = network
        elif w:
            self.data = _build_local_environment(self.data, self._groups, w)
            self.w = w
        elif distance:
            self.data = _build_local_environment(
                self.data, self._groups, bandwidth=distance, decay=decay
            )


def _build_local_environment(data, groups, w=None, bandwidth=None, decay="triangular"):
    """Convert observations into spatially-weighted sums.

    Parameters
    ----------
    data : DataFrame
        dataframe with local observations
    w : libpysal.weights object
        weights matrix defining the local environment

    Returns
    -------
    DataFrame
        Spatialized data

    """
    if not w:
        w = Kernel.from_dataframe(data, bandwidth=bandwidth, function=decay)
    new_data = []
    w = fill_diagonal(w)
    for y in data[groups]:
        new_data.append(lag_spatial(w, data[y]))
    new_data = pd.DataFrame(dict(zip(groups, new_data)))
    return new_data
