import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.spatial import SpatialProximity


class Spatial_Proximity_Tester(unittest.TestCase):
    def test_Spatial_Proximity(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP', 'TOT_POP']]
        index = SpatialProximity(df, 'HISP', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 1.0026623464135092)


if __name__ == '__main__':
    unittest.main()
