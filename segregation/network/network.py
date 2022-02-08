"""Calculate street network-based segregation measures."""

__author__ = "Elijah Knaap <elijah.knaap@ucr.edu> Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import pandas as pd
import os
import sys
from tqdm.auto import tqdm
import geopandas as gpd


# This class allows us to hide the diagnostic messages from urbanaccess if the `quiet` flag is set
class _HiddenPrints:  # from https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_osm_network(geodataframe, maxdist=5000, quiet=True, **kwargs):
    """Download a street network from OSM.

    Parameters
    ----------
    geodataframe : geopandas.GeoDataFrame
        geopandas.GeoDataFrame of the study area.
        Coordinate system should be in WGS84
    maxdist : int
        Total distance (in meters) of the network queries you may need.
        This is used to buffer the network to ensure theres enough to satisfy
        your largest query, otherwise there may be edge effects.
    quiet: bool
        If True, diagnostic messages from urbanaccess will be suppressed
    **kwargs : dict
        additional kwargs passed through to
        urbanaccess.ua_network_from_bbox

    Returns
    -------
    pandana.Network
        A pandana Network instance for use in accessibility calculations or
        spatial segregation measures that include a distance decay

    """
    try:
        import pandana as pdna
        from urbanaccess.osm.load import ua_network_from_bbox
    except ImportError:
        raise ImportError(
            "You need pandana and urbanaccess to work with segregation's network module\n"
            "You can install them with  `pip install urbanaccess pandana` "
            "or `conda install -c udst pandana urbanaccess`"
        )

    gdf = geodataframe.copy()
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdf = gdf.buffer(maxdist)
    bounds = gdf.to_crs(epsg=4326).total_bounds

    if quiet:
        print("Downloading data from OSM. This may take awhile.")
        with _HiddenPrints():
            net = ua_network_from_bbox(
                bounds[1], bounds[0], bounds[3], bounds[2], **kwargs
            )
    else:
        net = ua_network_from_bbox(bounds[1], bounds[0], bounds[3], bounds[2], **kwargs)
    print("Building network")
    network = pdna.Network(
        net[0]["x"], net[0]["y"], net[1]["from"], net[1]["to"], net[1][["distance"]]
    )

    return network


def calc_access(
    geodataframe,
    network,
    distance=2000,
    decay="linear",
    variables=None,
    precompute=True,
):
    """Calculate access to population groups.

    Parameters
    ----------
    geodataframe : geopandas.GeoDataFrame
        geodataframe with demographic data
    network : pandana.Network
        pandana.Network instance. This is likely created with `get_osm_network`
        or via helper functions from OSMnet or UrbanAccess.
    distance : int
        maximum distance to consider `accessible` (the default is 2000).
    decay : str
        decay type pandana should use "linear", "exp", or "flat"
        (which means no decay). The default is "linear".
    variables : list
        list of variable names present on gdf that should be calculated
    precompute: bool (default True)
        whether pandana should precompute the distance matrix. It can only be
        precomputed once, so If you plan to pass the same network to this
        function several times, you should set precompute=False for later runs

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns, `total_population` and `group_population`
        which represent the total number of each group that can be reached
        within the supplied `distance` parameter. The DataFrame is indexed
        on node_ids

    """
    if precompute:
        network.precompute(distance)

    geodataframe["node_ids"] = network.get_node_ids(
        geodataframe.centroid.x, geodataframe.centroid.y
    )

    access = []
    for variable in variables:
        network.set(
            geodataframe.node_ids, variable=geodataframe[variable], name=variable
        )

        access_pop = network.aggregate(distance, type="sum", decay=decay, name=variable)

        access.append(access_pop)
    names = ["acc_" + variable for variable in variables]
    access = pd.DataFrame(dict(zip(names, access)))

    return access


def compute_travel_cost_matrix(origins, destinations, network, reindex_name=None):
    """Compute a shortest path matrix from a pandana network

    Parameters
    ----------
    origins : geopandas.GeoDataFrame
        the set of origin geometries. If polygon input, the function will use their centroids
    destinations : geopandas.GeoDataFrame
        the set of destination geometries. If polygon input, the function will use their centroids
    network : pandana.Network
        Initialized pandana Network object holding a travel network for a study region
    reindex_name : str, optional
        Name of column on the origin/destinatation dataframe that holds unique index values
        If none (default), the index of the pandana Network node will be used

    Returns
    -------
    pandas.DataFrame
        an origin-destination cost matrix. Rows are origin indices, columns are destination indices,
        and values are shortest network path cost between the two
    """
    origins = origins.copy()
    destinations = destinations.copy()

    #  Note: these are not necessarily "OSM" ids, they're just the identifiers for each  node.
    #  with an integrated ped/transit network, these could be bus stops...
    origins["osm_ids"] = network.get_node_ids(origins.centroid.x, origins.centroid.y)

    destinations["osm_ids"] = network.get_node_ids(
        destinations.centroid.x, destinations.centroid.y
    )

    ods = {}

    with tqdm(total=len(origins["osm_ids"])) as pbar:
        for origin in origins["osm_ids"]:
            ods[f"{origin}"] = network.shortest_path_lengths(
                [origin] * len(origins), destinations["osm_ids"]
            )
            pbar.update(1)

    if reindex_name:
        df = pd.DataFrame(ods, index=origins[reindex_name])
        df.columns = df.index
    else:
        df = pd.DataFrame(ods, index=origins)

    return df


def reproject_network(network, crs):

    nodes = gpd.points_from_xy(network.nodes_df.x, network.nodes_df.y, crs=4326).to_crs(
        crs
    )
    network.nodes_df["x"] = nodes.x
    network.nodes_df["y"] = nodes.y

    return network
