import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.aspatial import ModifiedGiniSeg


class Modified_Gini_Seg_Tester(unittest.TestCase):

    def test_Modified_Gini_Seg(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        np.random.seed(1234)
        index = ModifiedGiniSeg(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.4217844443896344, decimal = 3)

if __name__ == '__main__':
    unittest.main()
