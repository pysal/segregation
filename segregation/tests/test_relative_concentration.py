import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.relative_concentration import Relative_Concentration


class Relative_Concentration_Tester(unittest.TestCase):

    def test_Relative_Concentration(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = Relative_Concentration(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.13102848628073688)

if __name__ == '__main__':
    unittest.main()
