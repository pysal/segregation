"""Calculate population accessibility."""

import pandas as pd
import pandana as pdna
from urbanaccess.osm.load import ua_network_from_bbox
from osmnx import project_gdf
import os
import sys


class _HiddenPrints:  # from https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_network(geodataframe, maxdist=5000, quiet=True, **kwargs):
    """Short summary.

    Parameters
    ----------
    geodataframe : geopandas.GeoDataFrame
        geopandas.GeoDataFrame of the study area.
    maxdist : int
        Total distance (in meters) of the network queries you may need
        this is used to buffer the network to ensure theres enough to satisfy
        your largest query.
    **kwargs : type
        Description of parameter `**kwargs`.

    Returns
    -------
    type
        Description of returned object.

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>>

    """
    gdf = geodataframe.copy()

    assert gdf.crs == {
        'init': 'epsg:4326'
    }, "geodataframe must be in epsg 4326"

    gdf = project_gdf(gdf)
    gdf = gdf.buffer(maxdist)
    bounds = gdf.to_crs(epsg=4326).total_bounds

    if quiet:
        print('Downloading data from OSM. This may take awhile.')
        with _HiddenPrints():
            net = ua_network_from_bbox(bounds[1], bounds[0], bounds[3],
                                       bounds[2], **kwargs)
    else:
        net = ua_network_from_bbox(bounds[1], bounds[0], bounds[3], bounds[2],
                                   **kwargs)
    print("Building network")
    network = pdna.Network(net[0]["x"], net[0]["y"], net[1]["from"],
                           net[1]["to"], net[1][["distance"]])

    return network


def calc_access(geodataframe,
                network,
                distance=5000,
                decay="linear",
                group_population=None,
                total_population=None):
    """Calculate access to population groups.

    Parameters
    ----------
    geodataframe : geopandas.GeoDataFrame
        geodataframe with demographic data
    network : pandana.Network
        pandana.Network instance. This is likely created with `get_network` or
        via helper functions from OSMnet or UrbanAccess.
    distance : int
        maximum distance to consider `accessible` (the default is 5000).
    decay : str
        decay type pandana should use "linear", "exponential" or "flat"
        (which means no decay). The default is "linear".
    group_population : str
        column name of the `group[_population` present on the input
        geodataframe.
    total_population : str
        column name of the `total_population` present on the input
        geodataframe.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns, `total_population` and `group_population`
        which represent the total number of each group that can be reached
        within the supplied `distance` parameter. The DataFrame is indexed
        on node_ids

    """
    network.precompute(distance)

    geodataframe["node_ids"] = network.get_node_ids(geodataframe.centroid.x,
                                                    geodataframe.centroid.y)

    network.set(geodataframe.node_ids,
                variable=geodataframe[group_population],
                name=group_population)

    network.set(geodataframe.node_ids,
                variable=geodataframe[total_population],
                name=total_population)

    access_total_pop = network.aggregate(distance,
                                         type="sum",
                                         decay=decay,
                                         name=str(total_population))

    access_group_pop = network.aggregate(distance,
                                         type="sum",
                                         decay=decay,
                                         name=str(group_population))

    access = pd.DataFrame({
        'total_population': access_total_pop,
        'group_population': access_group_pop
    })

    return access
