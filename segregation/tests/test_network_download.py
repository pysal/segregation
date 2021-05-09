from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.util import get_osm_network


def test_network_download():
    s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
    net = get_osm_network(s_map.iloc[[1]])
    assert net.edges_df.shape[0] >= 7942
