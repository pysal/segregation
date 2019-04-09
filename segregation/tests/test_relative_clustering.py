import unittest
import libpysal
import geopandas as gpd
import numpy as np
from segregation.spatial_indexes import Relative_Clustering


class Relative_Clustering_Tester(unittest.TestCase):

    def test_Relative_Clustering(self):
        s_map = gpd.read_file(libpysal.examples.get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = Relative_Clustering(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.009095632468738568)

if __name__ == '__main__':
    unittest.main()
