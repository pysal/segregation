import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.singlegroup import SpatialProximity


class Spatial_Proximity_Tester(unittest.TestCase):
    def test_Spatial_Proximity(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP', 'TOT_POP']]
        df = df.to_crs(df.estimate_utm_crs())
        index = SpatialProximity(df, 'HISP', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 1.0467806527410652)


if __name__ == '__main__':
    unittest.main()
