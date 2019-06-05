"""Calculate population accessibility."""

import pandas as pd
import geopandas as gpd
import pandana as pdna
#import osmnet
from urbanaccess.osm.load import ua_network_from_bbox
from osmnx import project_gdf
import os, sys


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
        print('Downloading data from OSM')
        with _HiddenPrints():
            net = ua_network_from_bbox(lng_min=bounds[0],
                                       lat_min=bounds[1],
                                       lng_max=bounds[2],
                                       lat_max=bounds[3],
                                       **kwargs)
    else:
        net = ua_network_from_bbox(lng_min=bounds[0],
                                   lat_min=bounds[1],
                                   lng_max=bounds[2],
                                   lat_max=bounds[3],
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
    """Short summary.

    Parameters
    ----------
    geodataframe : type
        Description of parameter `geodataframe`.
    network : type
        Description of parameter `network`.
    distance : type
        Description of parameter `distance` (the default is 5000).
    decay : type
        Description of parameter `decay` (the default is "linear").
    group_population : type
        Description of parameter `group_population` (the default is None).
    total_population : type
        Description of parameter `total_population` (the default is None).

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

    network.precompute(distance)

    gdf["node_ids"] = network.get_node_ids(gdf.x, gdf.y)
    network.set(gdf.node_ids,
                variable=gdf[group_population],
                name=group_population)

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
