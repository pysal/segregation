import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.singlegroup import Delta


class Delta_Tester(unittest.TestCase):
    def test_Delta(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP', 'TOT_POP']]
        index = Delta(df, 'HISP', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.8044969214141899)


if __name__ == '__main__':
    unittest.main()
