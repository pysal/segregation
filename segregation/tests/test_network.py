import unittest
import libpysal
import geopandas as gpd
from segregation.network import get_osm_network, calc_access


class Network_Tester(unittest.TestCase):
    def test_get_osm_network(self):
        df = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = df[df.FIPS.str.startswith('06061')]
        df = df[(df.centroid.x < -121) & (df.centroid.y < 38.85)]
        df.crs = {'init': 'epsg:4326'}
        test_net = get_osm_network(df, maxdist=0)
        assert len(test_net.edges_df) == 89018

    def test_calc_access(self):
        variables = ['WHITE_', 'BLACK_', 'ASIAN_', 'HISP_']
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['FIPS', 'geometry'] + variables]
        df = df[df.FIPS.str.startswith('06061')]
        df = df[(df.centroid.x < -121) & (df.centroid.y < 38.85)]
        df.crs = {'init': 'epsg:4326'}
        test_net = get_osm_network(df, maxdist=0)
        df[variables] = df[variables].astype(float)
        acc = calc_access(df, test_net, distance=1., variables=variables)
        assert acc.acc_WHITE_.sum() == 692010.0


if __name__ == '__main__':
    unittest.main()
