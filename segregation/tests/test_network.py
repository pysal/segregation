from libpysal.examples import load_example
import geopandas as gpd
from segregation.network import get_osm_network, calc_access


def test_calc_access():
    variables = ['WHITE', 'BLACK', 'ASIAN', 'HISP']
    s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
    df = s_map[['FIPS', 'geometry'] + variables]
    df = df[df.FIPS.str.startswith('06061')]
    df = df[(df.centroid.x < -121) & (df.centroid.y < 38.85)]
    df.crs = {'init': 'epsg:4326'}
    df[variables] = df[variables].astype(float)
    test_net = get_osm_network(df, maxdist=0)
    acc = calc_access(df, test_net, distance=1., variables=variables)
    assert acc.WHITE.sum() > 100
