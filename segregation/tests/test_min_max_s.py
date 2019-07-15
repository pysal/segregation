import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.spatial import MinMaxS


class Min_Max_S_Tester(unittest.TestCase):
    def test_Min_Max_S(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = MinMaxS(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.17119951092816454)


if __name__ == '__main__':
    unittest.main()
