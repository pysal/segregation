import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.singlegroup import ModifiedDissim


class Modified_Dissim_Tester(unittest.TestCase):
    def test_Modified_Dissim(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP', 'TOT_POP']]
        index = ModifiedDissim(df, 'HISP', 'TOT_POP', seed=1234, backend='loky')
        np.testing.assert_almost_equal(index.statistic, 0.31075891224250635, decimal = 3)


if __name__ == '__main__':
    unittest.main()
