import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.correlationr import Correlation_R


class Correlation_R_Tester(unittest.TestCase):

    def test_Correlation_R(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = Correlation_R(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.09164042012926693)

if __name__ == '__main__':
    unittest.main()
