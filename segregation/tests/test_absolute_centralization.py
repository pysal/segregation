import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.singlegroup import AbsoluteCentralization


class Absolute_Centralization_Tester(unittest.TestCase):
    def test_Absolute_Centralization(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP', 'TOT_POP']]
        index = AbsoluteCentralization(df, 'HISP', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.6891422368736286)


if __name__ == '__main__':
    unittest.main()
