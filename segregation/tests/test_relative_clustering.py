import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.singlegroup import RelativeClustering


class Relative_Clustering_Tester(unittest.TestCase):
    def test_Relative_Clustering(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP', 'TOT_POP']]
        index = RelativeClustering(df, 'HISP', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.652005507756501)


if __name__ == '__main__':
    unittest.main()
