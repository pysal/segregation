import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.aspatial import Gini_Seg


class Gini_Seg_Tester(unittest.TestCase):

    def test_Gini_Seg(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = Gini_Seg(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.43506510676886234)

if __name__ == '__main__':
    unittest.main()
