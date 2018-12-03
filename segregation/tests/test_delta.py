import unittest
import libpysal
import geopandas as gpd
import numpy as np
from inequality.delta import Delta


class Delta_Tester(unittest.TestCase):

    def test_Delta(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = Delta(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.8044969214141899)

if __name__ == '__main__':
    unittest.main()
