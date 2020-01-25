import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.aspatial import Dissim


class Dissim_Tester(unittest.TestCase):
    def test_Dissim(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP_', 'TOT_POP']]
        index = Dissim(df, 'HISP_', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.32184656076566864)


if __name__ == '__main__':
    unittest.main()
