import unittest
import libpysal
import geopandas as gpd
import numpy as np
from inequality.modified_dissimilarity import Modified_Dissim


class Modified_Dissim_Tester(unittest.TestCase):

    def test_Modified_Dissim(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        np.random.seed(1234)
        index = Modified_Dissim(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.31075891224250635)

if __name__ == '__main__':
    unittest.main()
