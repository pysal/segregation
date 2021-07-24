import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.singlegroup import DistanceDecayIsolation


class Distance_Decay_Isolation_Tester(unittest.TestCase):
    def test_Distance_Decay_Isolation(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP', 'TOT_POP']]
        df = df.to_crs(df.estimate_utm_crs())
        index = DistanceDecayIsolation(df, 'HISP', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.20144710843904595)


if __name__ == '__main__':
    unittest.main()
