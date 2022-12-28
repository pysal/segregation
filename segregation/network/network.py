"""Calculate street network-based segregation measures."""

__author__ = "Elijah Knaap <elijah.knaap@ucr.edu> Renan X. Cortes <renanc@ucr.edu> and Sergio J. Rey <sergio.rey@ucr.edu>"

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm


def _reproject_osm_nodes(nodes_df, input_crs, output_crs):
    #  take original x,y coordinates and convert into geopandas.Series, then reproject
    nodes = gpd.points_from_xy(x=nodes_df.x, y=nodes_df.y, crs=input_crs).to_crs(
        output_crs
    )
    #  convert to dataframe and recreate the x and y cols
    nodes = gpd.GeoDataFrame(index=nodes_df.index, geometry=nodes)
    nodes["x"] = nodes.centroid.x
    nodes["y"] = nodes.centroid.y
    return nodes


def calc_access(
    geodataframe,
    network,
    distance=2000,
    decay="linear",
    variables=None,
    precompute=True,
    return_node_data=False,
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
    return_node_data : bool, default is False
        Whether to return nodel-level accessibility data or to trim output to
        the same geometries as the input. Default is the latter.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns, `total_population` and `group_population`
        which represent the total number of each group that can be reached
        within the supplied `distance` parameter. The DataFrame is indexed
        on node_ids

    """
    if not decay:
        raise Exception("You must pass a decay function such as `linear`")
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
    access = pd.DataFrame(dict(zip(variables, access)))
    if return_node_data:
        return access.round(0)
    access = geodataframe[["node_ids", geodataframe.geometry.name]].merge(
        access, right_index=True, left_on="node_ids", how="left"
    )

    return access.dropna()


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


def project_network(network, output_crs=None, input_crs=4326):
    """Reproject a pandana.Network object into another coordinate system

    Parameters
    ----------
    network : pandana.Network
        an instantiated pandana Network object
    input_crs : int, optional
        the coordinate system used in the Network.node_df dataframe. Typically
        these data are collected in Lon/Lat, so the default 4326
    output_crs : int, str, or pyproj.crs.CRS, required
        EPSG code or pyproj.crs.CRS object of the output coordinate system

    Returns
    -------
    pandana.Network
        an initialized pandana.Network with 'x' and y' values represented
        by coordinates in the specified CRS
    """
    try:
        import pandana as pdna
    except ImportError:
        raise ImportError(
            "You need pandana to work with segregation's network module\n"
            "You can install them with  `pip install urbanaccess pandana` "
            "or `conda install -c udst pandana urbanaccess`"
        )

    assert output_crs, "You must provide an output CRS"

    #  take original x,y coordinates and convert into geopandas.Series, then reproject
    nodes = _reproject_osm_nodes(network.nodes_df, input_crs, output_crs)

    #  reinstantiate the network (needs to rebuild the tree)
    net = pdna.Network(
        node_x=nodes["x"],
        node_y=nodes["y"],
        edge_from=network.edges_df["from"],
        edge_to=network.edges_df["to"],
        edge_weights=network.edges_df[network.impedance_names],
        twoway=network._twoway,
    )
    return net
