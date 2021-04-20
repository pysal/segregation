import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
from segregation.singlegroup import BiasCorrectedDissim


class Bias_Corrected_Dissim_Tester(unittest.TestCase):
    def test_Bias_Corrected_Dissim(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP', 'TOT_POP']]
        np.random.seed(1234)
        index = BiasCorrectedDissim(df, 'HISP', 'TOT_POP')
        np.testing.assert_almost_equal(index.statistic, 0.32136474449360836, decimal = 3)


if __name__ == '__main__':
    unittest.main()
