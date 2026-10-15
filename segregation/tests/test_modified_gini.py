import unittest
from libpysal.examples import load_example
import geopandas as gpd
import numpy as np
import pandas as pd
from segregation.singlegroup import ModifiedGini


class Modified_Gini_Seg_Tester(unittest.TestCase):

    def test_Modified_Gini(self):
        s_map = gpd.read_file(load_example("Sacramento1").get_path("sacramentot2.shp"))
        df = s_map[['geometry', 'HISP', 'TOT_POP']]
        index = ModifiedGini(df, 'HISP', 'TOT_POP', seed=1234, backend='loky')
        np.testing.assert_almost_equal(index.statistic, 0.4217904415805389, decimal=5)
        again = ModifiedGini(df, 'HISP', 'TOT_POP', seed=1234, backend='loky')
        self.assertEqual(index.statistic, again.statistic)
        zero = ModifiedGini(
            df, 'HISP', 'TOT_POP', seed=0, iterations=2, backend='loky', n_jobs=1
        )
        zero_again = ModifiedGini(
            df, 'HISP', 'TOT_POP', seed=0, iterations=2, backend='loky', n_jobs=1
        )
        self.assertEqual(zero.statistic, zero_again.statistic)
        missing = ModifiedGini(
            df, 'HISP', 'TOT_POP', iterations=2, backend='loky', n_jobs=1
        )
        self.assertIsInstance(missing.statistic, float)

    def test_iterations_draw_more_than_once(self):
        small = pd.DataFrame({"g": [10, 20, 30, 40], "t": [100, 100, 100, 100]})
        two = ModifiedGini(
            small, "g", "t", seed=1234, iterations=2, backend="threading", n_jobs=1
        )
        eight = ModifiedGini(
            small, "g", "t", seed=1234, iterations=8, backend="threading", n_jobs=1
        )
        eight_again = ModifiedGini(
            small, "g", "t", seed=1234, iterations=8, backend="threading", n_jobs=1
        )
        np.testing.assert_almost_equal(two.statistic, 0.2581572150885745)
        np.testing.assert_almost_equal(eight.statistic, 0.2533236196830427)
        self.assertEqual(eight.statistic, eight_again.statistic)

if __name__ == '__main__':
    unittest.main()
