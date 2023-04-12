"""Base classes for segregation indices."""

from warnings import warn
import warnings

import geopandas as gpd
import libpysal
import pandas as pd
from libpysal.weights import lag_spatial
from libpysal.weights.distance import Kernel
from libpysal.weights.util import attach_islands, fill_diagonal

from .network import calc_access
from .util.util import _nan_handle


def _return_length_weighted_w(data):
    """
    Returns a PySAL weights object that the weights represent the length of the common boundary of two areal units that share border.
    Author: Levi Wolf <levi.john.wolf@gmail.com>.
    Thank you, Levi!

    Parameters
    ----------

    data          : a geopandas DataFrame with a 'geometry' column.

    Notes
    -----
    Currently it's not making any projection.

    """

    w = libpysal.weights.Rook.from_dataframe(
        data, ids=data.index.tolist(), geom_col=data._geometry_column_name
    )

    if len(w.islands) == 0:
        w = w
    else:
        warnings.warn("There are some islands in the GeoDataFrame.")
        w_aux = libpysal.weights.KNN.from_dataframe(
            data, ids=data.index.tolist(), geom_col=data._geometry_column_name, k=1
        )
        w = attach_islands(w, w_aux)

    adjlist = w.to_adjlist()
    islands = pd.DataFrame.from_records(
        [{"focal": island, "neighbor": island, "weight": 0} for island in w.islands]
    )
    merged = adjlist.merge(
        data.geometry.to_frame("geometry"),
        left_on="focal",
        right_index=True,
        how="left",
    ).merge(
        data.geometry.to_frame("geometry"),
        left_on="neighbor",
        right_index=True,
        how="left",
        suffixes=("_focal", "_neighbor"),
    )
    # Transforming from pandas to geopandas
    merged = gpd.GeoDataFrame(merged, geometry="geometry_focal")
    merged["geometry_neighbor"] = gpd.GeoSeries(merged.geometry_neighbor)

    # Getting the shared boundaries
    merged["shared_boundary"] = merged.geometry_focal.intersection(
        merged.set_geometry("geometry_neighbor")
    )

    # Putting it back to a matrix
    merged["weight"] = merged.set_geometry("shared_boundary").length
    merged_with_islands = pd.concat((merged, islands))
    length_weighted_w = libpysal.weights.W.from_adjlist(
        merged_with_islands[["focal", "neighbor", "weight"]]
    )
    for island in w.islands:
        length_weighted_w.neighbors[island] = []
        del length_weighted_w.weights[island]

    length_weighted_w._reset()

    return length_weighted_w


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

        if any([type(group_pop_var) is not str, type(total_pop_var) is not str]):
            raise TypeError("group_pop_var and total_pop_var must be strings")

        if any([group_pop_var not in data.columns, total_pop_var not in data.columns]):
            raise ValueError(
                "group_pop_var and total_pop_var columns must be present on the dataframe"
            )

        if any(data[total_pop_var] < data[group_pop_var]):
            raise ValueError(
                "Group of interest population must equal or lower than the total population of the units."
            )

        if isinstance(data, gpd.GeoDataFrame):
            data = _nan_handle(
                data[[group_pop_var, total_pop_var, data.geometry.name]]
            )
            data = data[[group_pop_var, total_pop_var, data.geometry.name]]

        else:
            data = _nan_handle(data[[group_pop_var, total_pop_var]])
            data =data[[group_pop_var, total_pop_var]]

        data["group_2_pop_var"] = data[total_pop_var] - data[group_pop_var]

        self.index_type = "singlegroup"
        self.data = data.copy()
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
        self.data = data.copy()
        self.groups = groups


class SpatialExplicitIndex:
    """Class for estimating segregation indices that are explicitly spatial (have no aspatial version)."""

    def __init__(self, **kwargs):
        """Initialize spatially explicit index."""
        if not isinstance(self.data, gpd.GeoDataFrame):
            raise TypeError(
                "`data` must be a geopanads.GeoDataFrame with a vaild geometry column"
            )
        if self.data.crs.is_geographic:
            warn(
                "Geometry is in a geographic CRS. Distance and area calculations in this index are likely incorrect. "
                "Re-project the input data to a projected CRS using `GeoDataFrame.to_crs()` before calculating this index."
            )
        self.spatial_type = "explicit"


class SpatialImplicitIndex:
    """Class for estimating segregation indices that can be spatial or aspatial."""

    def __init__(
        self,
        w,
        network,
        distance=1000,
        decay="linear",
        function="triangular",
        precompute=False,
        **kwargs
    ):
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
            type of decay function to apply (passed to `pandana`). Options include
            {'linear', 'exponential', or 'flat'}
        function : str
            decay function to use in spatial weights object (passed to libpysal.weights.Kernel)
            options = {'triangular','uniform','quadratic','quartic','gaussian'}
        precompute : bool
            Whether to precompute the pandana Network object
        """
        self.spatial_type = "implicit"
        if self.index_type == "multigroup":
            self._groups = self.groups
        elif self.index_type == "singlegroup":
            self._groups = [self.group_pop_var, self.total_pop_var, "group_2_pop_var"]

        if w and network:
            raise AttributeError(
                "must pass either a pandana network or a pysal weights object, but not both"
            )
        if network:
            access = calc_access(
                self.data,
                variables=self._groups,
                network=network,
                distance=distance,
                decay=decay,
                precompute=precompute,
            )
            self._original_data = self.data.copy()
            self.data = access
            self.network = network
        elif w:
            self.data = _build_local_environment(
                self.data, self._groups, w, function=function
            )
            self.w = w
        elif distance and not network:
            self._original_data = self.data.copy()
            self.data = _build_local_environment(
                self.data, self._groups, bandwidth=distance, function=function
            )


def _build_local_environment(
    data, groups, w=None, bandwidth=1000, function="triangular"
):
    """Convert observations into spatially-weighted sums.

    Parameters
    ----------
    data : GeoDataFrame
        dataframe with local observations
    w : libpysal.weights object
        weights matrix defining the local environment

    Returns
    -------
    DataFrame
        Spatialized data

    """
    data = data.copy().reset_index()
    if data.crs.is_geographic:
        warnings.warn(
            "GeoDataFrame appears to have a geographic coordinate system and likely needs to be reprojected"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not w:
            w = Kernel.from_dataframe(data, bandwidth=bandwidth, function=function)
        new_data = []
        w = fill_diagonal(w)
        for y in data[groups]:
            new_data.append(lag_spatial(w, data[y]))
        new_data = pd.DataFrame(dict(zip(groups, new_data))).round(0)
        new_data = data.geometry.to_frame().join(new_data.reset_index())

        return new_data
