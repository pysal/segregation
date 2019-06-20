import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.spatial import Distance_Decay_Exposure


class Distance_Decay_Exposure_Tester(unittest.TestCase):
    def test_Distance_Decay_Exposure(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = Distance_Decay_Exposure(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.8396583368412371)


if __name__ == '__main__':
    unittest.main()
